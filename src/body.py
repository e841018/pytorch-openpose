import numpy as np, torch
import torchvision.transforms as T, torchvision.transforms.functional as TF
from . import util, model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Body(object):
    def __init__(self, model_path):
        self.model = model.bodypose_model().to(device)
        model_weights = torch.load(model_path, map_location=device)
        state_dict = model.transfer(model_weights, self.model.state_dict().keys())
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def infer(self, img0, scales=[0.7, 1.0, 1.3]):
        B, C, H0, W0 = img0.shape
        assert img0.dtype == torch.float32

        # the model has 3 nn.MaxPool2d(stride=2) layers
        stride = 2 ** 3

        n_channel = util.n_limb*2 + util.n_keypoint + 1
        out6_mean = torch.zeros((B, n_channel, H0, W0), dtype=torch.float32, device=img0.device)
        n_scale = len(scales)
        for scale in scales:
            # scale
            H1 = int(scale * 368)
            W1 = int(H1 / H0 * W0)
            img1 = TF.resize(img0, (H1, W1), interpolation=T.InterpolationMode.BICUBIC)
            # pad
            img2 = torch.nn.functional.pad(img1, (0, -W1%stride, 0, -H1%stride), value=0.5)
            H2, W2 = img2.shape[-2:]
            # main model
            out6 = self.model(img2) # shape=(B, n_channel, H2/stride, W2/stride)
            # undo nn.MaxPool2d downsampling
            out6 = TF.resize(out6, (H2, W2), interpolation=T.InterpolationMode.BICUBIC)
            # unpad
            out6 = out6[:, :, :H1, :W1]
            # undo scale
            out6 = TF.resize(out6, (H0, W0), interpolation=T.InterpolationMode.BICUBIC)
            out6_mean += out6
        out6_mean /= n_scale
        paf = out6[:, :util.n_limb*2]
        heatmap = out6[:, util.n_limb*2:-1]
        # the last heatmap channel is background, not used here
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/80d4c5f7b25ba4c3bf5745ab7d0e6ccd3db8b242/src/openpose/pose/poseParameters.cpp#L54
        return paf, heatmap

    def __call__(self, img, sigma=3.0, thre1=0.1, n_linspace=10, thre2=0.05):

        ## check shape
        H, W, C = img.shape
        assert C == 3, C

        ## convert to tensor
        img = torch.tensor(img.copy(), dtype=torch.float32, device=device) / 255 - 0.5
        img = img.moveaxis(-1, 0)[None, :, :, :]

        ## tensor operations
        with torch.no_grad():
            paf, heatmap = self.infer(img)
            heatmap_blur = TF.gaussian_blur(heatmap,
                kernel_size=int(np.ceil(sigma * 4)) * 2 + 1,
                sigma=sigma)

        ## convert to ndarray
        paf          = paf         .detach().cpu().numpy()[0] # shape=(n_limb*2, H, W)
        heatmap      = heatmap     .detach().cpu().numpy()[0] # shape=(n_keypoint, H, W)
        heatmap_blur = heatmap_blur.detach().cpu().numpy()[0] # shape=(n_keypoint, H, W)

        ## find peaks for each keypoint
        all_peaks = [] # all_peaks[k][p] = (x, y, score, peak_id)
        peak_counter = 0
        for k in range(util.n_keypoint):
            map_blur = heatmap_blur[k]
            map_blur_pad = np.pad(map_blur, 1)
            peaks_binary = np.logical_and.reduce((
                map_blur > thre1,
                map_blur >= map_blur_pad[1:-1, :-2], # left
                map_blur >= map_blur_pad[1:-1, 2:],  # right
                map_blur >= map_blur_pad[:-2, 1:-1], # top
                map_blur >= map_blur_pad[2:, 1:-1],  # bottom
            ))
            peaks = []
            for y, x in zip(*np.nonzero(peaks_binary)):
                score = heatmap[k, y, x]
                peak_id = peak_counter
                peak_counter += 1
                peaks.append((x, y, score, peak_id))
            all_peaks.append(peaks)

        ## find connections for each limb
        all_connections = [] # all_connections[l][c] = (peak_id0, peak_id1, paf_score)
        for l in range(util.n_limb):
            k0, k1 = util.l2k[l]
            peaks0, peaks1 = all_peaks[k0], all_peaks[k1]
            max_n_connection = min(len(peaks0), len(peaks1))
            if len(peaks0) == 0 or len(peaks1) == 0:
                all_connections.append(np.array([]))
                continue
            paf_x, paf_y  = paf[util.l2m[l]]
            connections_cand = []
            for x0, y0, score0, peak_id0 in peaks0:
                for x1, y1, score1, peak_id1 in peaks1:
                    samp_x = np.linspace(x0, x1, n_linspace).round().astype(int)
                    samp_y = np.linspace(y0, y1, n_linspace).round().astype(int)
                    vec = np.array((x1-x0, y1-y0), dtype=np.float32)
                    vec_norm = max(0.001, (vec ** 2).sum() ** 0.5)
                    vec /= vec_norm
                    paf_vec = paf_x[samp_y, samp_x] * vec[0] \
                            + paf_y[samp_y, samp_x] * vec[1]
                    dist_prior = min(0.5 * H / vec_norm - 1, 0)
                    paf_score = paf_vec.mean() + dist_prior
                    if (paf_vec > thre2).mean() > 0.8 and paf_score > 0:
                        connections_cand.append((peak_id0, peak_id1, paf_score))
            connections_cand.sort(key=lambda conn: conn[2], reverse=True)

            peak_id_used = set()
            connections = []
            for connection in connections_cand:
                peak_id0, peak_id1 = connection[:2]
                if peak_id0 in peak_id_used or peak_id1 in peak_id_used:
                    continue
                connections.append(connection)
                peak_id_used.add(peak_id0)
                peak_id_used.add(peak_id1)
                if len(connections) >= max_n_connection:
                    break
            all_connections.append(np.array(connections))

        ## collect list of all peaks
        # peak_list: shape=(n_peak, 4)
        # peak_list[peak_id] = x, y, score, peak_id
        peak_list = []
        for peaks in all_peaks:
            peak_list += peaks
        peak_list = np.array(peak_list, dtype=np.float32)

        ## cluster connections into persons
        # persons: shape=(n_person, n_keypoint+2)
        # persons[p][:-2] are the peak_id of each keypoint, -1 if not present
        # persons[p][-2] is the sum of (1) score of the keypoints and (2) paf_score of the limbs
        # persons[p][-1] is the number of keypoints
        persons = []
        for l, connections in enumerate(all_connections):
            if len(connections) == 0:
                continue
            k0, k1 = util.l2k[l]
            for peak_id0, peak_id1, paf_score in connections:
                peak_id0 = int(peak_id0)
                peak_id1 = int(peak_id1)

                # look for existing persons that is matched to either end of the current connection
                person0 = None
                person1 = None
                for person in persons:
                    if person0 is not None and person1 is not None:
                        break
                    if person[k0] == peak_id0:
                        person0 = person
                    if person[k1] == peak_id1:
                        person1 = person
                n_found = (person0 is not None) + (person1 is not None)

                # case (0): no matched, create a new person
                if n_found == 0:
                    if l < 17: # not considering shoulder-ear connections
                        person = np.ones(20, dtype=np.float32) * -1
                        person[k0] = peak_id0
                        person[k1] = peak_id1
                        person[-2] = paf_score + peak_list[peak_id0, 2] + peak_list[peak_id1, 2]
                        person[-1] = 2
                        persons.append(person)

                # case (1): 1 end matched
                elif n_found == 1:
                    if person0 is not None:
                        person0[k1] = peak_id1
                        person0[-2] += paf_score + peak_list[peak_id1, 2]
                        person0[-1] += 1
                    if person1 is not None:
                        person1[k0] = peak_id0
                        person1[-2] += paf_score + peak_list[peak_id0, 2]
                        person1[-1] += 1

                # case (2): 2 ends matched
                else:
                    # case (2.0): if they are the same person (loop), add paf_score
                    if id(person0) == id(person1):
                        person0[-2] += paf_score
                    # case (2.1): if they are disjoint persons, merge them
                    elif np.all(person0[:-2]==-1 | person1[:-2]==-1):
                        person0 += person1
                        person0[:-2] += 1
                        person0[-2] += paf_score
                        for p, person in enumerate(persons):
                            if id(person) == id(person1):
                                del persons[p]
                                break
                    # case (2.2): if they have one or more keypoints matched to different peaks, there's a conflict
                    # attribute the current connection to the person with the higher score
                    else:
                        if person0[-2] >= person1[-2]:
                            person0[k1] = peak_id1
                            person0[-2] += paf_score + peak_list[peak_id1, 2]
                            person0[-1] += 1
                        else:
                            person1[k0] = peak_id0
                            person1[-2] += paf_score + peak_list[peak_id0, 2]
                            person1[-1] += 1

        ## delete the persons with few keypoints
        to_delete = []
        for p, person in enumerate(persons):
            if person[-1] < 4 or person[-2] / person[-1] < 0.4:
                to_delete.append(p)
        for p in to_delete[::-1]:
            del persons[p]
        persons = np.array(persons, dtype=np.float32)

        return peak_list, persons

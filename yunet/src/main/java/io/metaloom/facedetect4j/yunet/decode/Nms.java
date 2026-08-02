package io.metaloom.facedetect4j.yunet.decode;

import java.util.ArrayList;
import java.util.List;

import io.metaloom.facedetect4j.api.Face;

/** Greedy IoU non-maximum suppression. */
public final class Nms {

	private Nms() {
	}

	/**
	 * @param faces  <b>already sorted by score descending</b>
	 * @param thresh IoU above which the lower-scoring box is dropped
	 */
	public static List<Face> greedy(List<Face> faces, float thresh) {
		int n = faces.size();
		boolean[] dead = new boolean[n];
		List<Face> keep = new ArrayList<>();
		for (int i = 0; i < n; i++) {
			if (dead[i]) {
				continue;
			}
			Face a = faces.get(i);
			keep.add(a);
			for (int j = i + 1; j < n; j++) {
				if (!dead[j] && a.box().iou(faces.get(j).box()) > thresh) {
					dead[j] = true;
				}
			}
		}
		return keep;
	}
}

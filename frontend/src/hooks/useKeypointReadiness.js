import { useEffect, useMemo, useState } from "react";
import { getMockKeypointState } from "../mocks/mockKeypointDetector";

const EMPTY_KEYPOINTS = {
  hands: false,
  face: false,
  pose: false,
};

export function useKeypointReadiness(active) {
  const [keypoints, setKeypoints] = useState(EMPTY_KEYPOINTS);

  useEffect(() => {
    if (!active) {
      setKeypoints(EMPTY_KEYPOINTS);
      return undefined;
    }

    let step = 0;
    setKeypoints(EMPTY_KEYPOINTS);

    const timer = window.setInterval(() => {
      step += 1;
      setKeypoints(getMockKeypointState(step));
    }, 650);

    return () => window.clearInterval(timer);
  }, [active]);

  const isReady = useMemo(
    () => keypoints.hands && keypoints.face && keypoints.pose,
    [keypoints.face, keypoints.hands, keypoints.pose],
  );

  return {
    keypoints,
    isReady,
  };
}

export function getMockKeypointState(step) {
  return {
    hands: step >= 1,
    face: step >= 2,
    pose: step >= 3,
  };
}

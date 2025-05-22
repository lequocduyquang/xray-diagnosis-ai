export function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map((v) => v / sum);
}

export function getPredictedClass(probabilities) {
  let maxIndex = 0;
  probabilities.forEach((p, i) => {
    if (p > probabilities[maxIndex]) maxIndex = i;
  });
  return maxIndex;
}

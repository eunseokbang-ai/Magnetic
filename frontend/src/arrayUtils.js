// Math.min(...arr)/Math.max(...arr) blow the call stack once arr has more
// than roughly 60-120k elements (spreading builds one giant argument list) -
// a real risk here since a multi-file upload can produce hundreds of
// thousands of points. Plain loops are O(n) with no such limit.
export function minMax(values) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return [min, max];
}

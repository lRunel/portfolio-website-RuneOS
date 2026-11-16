// figures.js — mounts hairline line-figures (MIT, @lucasmarkes/hairline) into [data-figure] elements.
import * as hairline from '../js/vendor/hairline.js';

document.querySelectorAll('[data-figure]').forEach((el) => {
  const make = hairline[el.dataset.figure];
  if (make) make(el, { theme: 'dark', intensity: 0.8, play: true });
});

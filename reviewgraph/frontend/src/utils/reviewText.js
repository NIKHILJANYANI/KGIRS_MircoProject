export function normalizeReviewText(value) {
  return String(value ?? '').replace(/<br\s*\/?>/gi, '\n')
}

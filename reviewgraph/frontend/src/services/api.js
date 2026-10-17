import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
})

// Add response interceptor for error handling
api.interceptors.response.use(
  response => response.data,
  error => {
    const message = error.response?.data?.detail || error.message || 'Unknown error'
    return Promise.reject(new Error(message))
  }
)

// Health and stats
export const getHealth = () => api.get('/health')
export const getStats  = () => api.get('/stats')

// Products
export const getProducts             = (params) => api.get('/products', { params })
export const getProduct              = (id)     => api.get(`/products/${id}`)
export const investigateProduct      = (id)     => api.get(`/products/${id}/investigation`)
export const getProductInvestigation = (id)     => api.get(`/products/${id}/investigation`)
export const getProductGraph         = (id)     => api.get(`/products/${id}/graph`)

// Reviews
export const getReview         = (id) => api.get(`/reviews/${id}`)
export const getRelatedReviews = (id) => api.get(`/reviews/${id}/related`)

// Search
export const search        = (payload) => api.post('/search', payload)
export const searchReviews = (payload) => api.post('/search', payload)

// Evaluation
export const getEvaluation = () => api.get('/evaluation')

export default api

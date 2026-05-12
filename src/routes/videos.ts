// Video Upload Routes - Secure Storage and Playback

import { Hono } from 'hono'
import type { Bindings, Variables } from '../types.js'
import { authMiddleware } from '../middleware/auth.js'
import { getPool } from '../database.js'

const videos = new Hono<{ Bindings: Bindings; Variables: Variables }>()

// Constants
const MAX_VIDEO_SIZE = 500 * 1024 * 1024 // 500MB max
const ALLOWED_TYPES = ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo']

// Apply auth middleware
videos.use('*', authMiddleware)

// POST /api/videos/upload - Upload a video
videos.post('/upload', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const formData = await c.req.formData()
    const videoFile = formData.get('video') as File
    const patientId = parseInt(formData.get('patient_id') as string)
    const videoType = formData.get('video_type') as string || 'assessment'
    const title = formData.get('title') as string
    const description = formData.get('description') as string
    const assessmentId = formData.get('assessment_id') ? parseInt(formData.get('assessment_id') as string) : null

    if (!videoFile) {
      return c.json({ success: false, error: 'No video file provided' }, 400)
    }

    if (!patientId || isNaN(patientId)) {
      return c.json({ success: false, error: 'Valid patient_id is required' }, 400)
    }

    // Validate file type
    if (!ALLOWED_TYPES.includes(videoFile.type)) {
      return c.json({ 
        success: false, 
        error: `Invalid file type: ${videoFile.type}. Allowed: ${ALLOWED_TYPES.join(', ')}` 
      }, 400)
    }

    // Validate file size
    if (videoFile.size > MAX_VIDEO_SIZE) {
      return c.json({ 
        success: false, 
        error: `File too large. Max size: ${MAX_VIDEO_SIZE / 1024 / 1024}MB` 
      }, 413)
    }

    // Get clinician info
    const clinician = c.get('clinician')

    // Generate unique storage key
    const timestamp = Date.now()
    const fileExtension = videoFile.name.split('.').pop()
    const storageKey = `videos/${patientId}/${videoType}/${timestamp}_${Math.random().toString(36).substring(7)}.${fileExtension}`

    // Note: Video storage is currently metadata-only on Railway. 
    // In production, integrate with S3 or local persistent storage.
    let storageUrl = null

    // Save video metadata to database
    const result = await pool.query(
      `INSERT INTO videos (
        patient_id, uploaded_by, video_type, title, description,
        storage_provider, storage_key, storage_url,
        file_size_bytes, duration_seconds, resolution,
        assessment_id, processing_status, created_at
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW())
      RETURNING *`,
      [
        patientId,
        clinician?.id || null,
        videoType,
        title || videoFile.name,
        description || null,
        'database',
        storageKey,
        storageUrl,
        videoFile.size,
        null, // duration
        null, // resolution
        assessmentId,
        'completed'
      ]
    )

    return c.json({
      success: true,
      data: result.rows[0],
      message: 'Video uploaded successfully'
    })

  } catch (error: any) {
    console.error('[VIDEO] Upload error:', error.message)
    return c.json({ success: false, error: 'Failed to upload video' }, 500)
  }
})

// GET /api/videos - List videos for a patient
videos.get('/', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  try {
    const patientId = c.req.query('patient_id')
    const videoType = c.req.query('type')

    let whereClause = 'WHERE 1=1'
    const params: any[] = []

    if (patientId) {
      params.push(parseInt(patientId))
      whereClause += ` AND v.patient_id = $${params.length}`
    }

    if (videoType) {
      params.push(videoType)
      whereClause += ` AND v.video_type = $${params.length}`
    }

    const result = await pool.query(
      `SELECT v.*, 
              p.first_name as patient_first_name, 
              p.last_name as patient_last_name,
              u.first_name as uploaded_by_first_name,
              u.last_name as uploaded_by_last_name
       FROM videos v
       LEFT JOIN patients p ON v.patient_id = p.id
       LEFT JOIN users u ON v.uploaded_by = u.id
       ${whereClause}
       ORDER BY v.created_at DESC`,
      params
    )

    return c.json({
      success: true,
      data: result.rows
    })

  } catch (error: any) {
    console.error('[VIDEO] List error:', error.message)
    return c.json({ success: false, error: 'Failed to load videos' }, 500)
  }
})

// GET /api/videos/:id - Get video metadata and playback URL
videos.get('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid video ID' }, 400)
  }

  try {
    const result = await pool.query(
      `SELECT v.*, 
              p.first_name as patient_first_name, 
              p.last_name as patient_last_name
       FROM videos v
       LEFT JOIN patients p ON v.patient_id = p.id
       WHERE v.id = $1`,
      [id]
    )

    if (result.rows.length === 0) {
      return c.json({ success: false, error: 'Video not found' }, 404)
    }

    const video = result.rows[0]
    const playbackUrl = video.storage_url

    return c.json({
      success: true,
      data: {
        ...video,
        playback_url: playbackUrl
      }
    })

  } catch (error: any) {
    console.error('[VIDEO] Get error:', error.message)
    return c.json({ success: false, error: 'Failed to load video' }, 500)
  }
})

// DELETE /api/videos/:id - Delete video
videos.delete('/:id', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid video ID' }, 400)
  }

  try {
    // Get video info first
    const videoResult = await pool.query(
      'SELECT storage_provider, storage_key FROM videos WHERE id = $1',
      [id]
    )

    if (videoResult.rows.length === 0) {
      return c.json({ success: false, error: 'Video not found' }, 404)
    }

    const video = videoResult.rows[0]

    // Delete from database
    await pool.query('DELETE FROM videos WHERE id = $1', [id])

    return c.json({
      success: true,
      message: 'Video deleted successfully'
    })

  } catch (error: any) {
    console.error('[VIDEO] Delete error:', error.message)
    return c.json({ success: false, error: 'Failed to delete video' }, 500)
  }
})

// POST /api/videos/:id/process - Trigger video processing (transcoding, thumbnail)
videos.post('/:id/process', async (c) => {
  const pool = getPool()
  if (!pool) {
    return c.json({ success: false, error: 'Database not available' }, 503)
  }

  const id = parseInt(c.req.param('id'))
  if (isNaN(id)) {
    return c.json({ success: false, error: 'Invalid video ID' }, 400)
  }

  try {
    // Update processing status
    await pool.query(
      "UPDATE videos SET processing_status = 'processing' WHERE id = $1",
      [id]
    )

    // In production, this would queue a background job for processing
    // For now, we just mark it as completed
    await pool.query(
      "UPDATE videos SET processing_status = 'completed' WHERE id = $1",
      [id]
    )

    return c.json({
      success: true,
      message: 'Video processing started'
    })

  } catch (error: any) {
    console.error('[VIDEO] Process error:', error.message)
    return c.json({ success: false, error: 'Failed to process video' }, 500)
  }
})

export default videos

/**
 * PhysioMotion - Global Notification System
 * Provides user feedback for success, error, warning, and info messages
 */

(function() {
  // Inject notification container into DOM
  const notificationContainer = document.createElement('div');
  notificationContainer.id = 'notification-container';
  notificationContainer.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    pointer-events: none;
  `;
  
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      document.body.appendChild(notificationContainer);
    });
  } else {
    document.body.appendChild(notificationContainer);
  }

  // Inject notification styles
  const style = document.createElement('style');
  style.textContent = `
    .notification {
      min-width: 300px;
      max-width: 500px;
      padding: 16px 24px;
      margin-bottom: 12px;
      border-radius: 8px;
      background: white;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(0, 0, 0, 0.05);
      transform: translateX(450px);
      transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55), opacity 0.3s ease;
      pointer-events: all;
      display: flex;
      align-items: center;
      gap: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      opacity: 0;
    }
    
    .notification.show {
      transform: translateX(0);
      opacity: 1;
    }
    
    .notification-icon {
      flex-shrink: 0;
      width: 20px;
      height: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
    }
    
    .notification-content {
      flex: 1;
      color: #1f2937;
    }
    
    .notification-close {
      flex-shrink: 0;
      width: 20px;
      height: 20px;
      border: none;
      background: none;
      cursor: pointer;
      opacity: 0.5;
      transition: opacity 0.2s;
      font-size: 18px;
      line-height: 1;
      padding: 0;
      color: #6b7280;
    }
    
    .notification-close:hover {
      opacity: 1;
    }
    
    .notification-success {
      border-left: 4px solid #10b981;
    }
    
    .notification-success .notification-icon {
      color: #10b981;
    }
    
    .notification-error {
      border-left: 4px solid #ef4444;
    }
    
    .notification-error .notification-icon {
      color: #ef4444;
    }
    
    .notification-warning {
      border-left: 4px solid #f59e0b;
    }
    
    .notification-warning .notification-icon {
      color: #f59e0b;
    }
    
    .notification-info {
      border-left: 4px solid #3b82f6;
    }
    
    .notification-info .notification-icon {
      color: #3b82f6;
    }
    
    @media (max-width: 640px) {
      #notification-container {
        left: 20px;
        right: 20px;
        top: 20px;
      }
      
      .notification {
        min-width: auto;
        width: 100%;
      }
    }
  `;
  document.head.appendChild(style);

  /**
   * Show a notification
   * @param {string} message - The message to display
   * @param {string} type - The type of notification: 'success', 'error', 'warning', or 'info'
   * @param {number} duration - Duration in milliseconds (default: 4000)
   * @returns {HTMLElement} The notification element
   */
  window.showNotification = function(message, type = 'info', duration = 4000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Icon based on type
    const icons = {
      success: '✓',
      error: '✕',
      warning: '⚠',
      info: 'ℹ'
    };
    
    notification.innerHTML = `
      <div class="notification-icon">${icons[type] || icons.info}</div>
      <div class="notification-content">${escapeHtml(message)}</div>
      <button class="notification-close" aria-label="Close notification">×</button>
    `;
    
    // Add to container
    const container = document.getElementById('notification-container');
    if (!container) {
      console.error('Notification container not found');
      return null;
    }
    
    container.appendChild(notification);
    
    // Trigger animation
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        notification.classList.add('show');
      });
    });
    
    // Close button handler
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
      hideNotification(notification);
    });
    
    // Auto-hide after duration
    if (duration > 0) {
      setTimeout(() => {
        hideNotification(notification);
      }, duration);
    }
    
    return notification;
  };

  /**
   * Hide a notification
   * @param {HTMLElement} notification - The notification element to hide
   */
  function hideNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 300);
  }

  /**
   * Escape HTML to prevent XSS
   * @param {string} text - Text to escape
   * @returns {string} Escaped text
   */
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  /**
   * Show success notification
   * @param {string} message - Success message
   * @param {number} duration - Duration in milliseconds
   */
  window.showSuccess = function(message, duration) {
    return showNotification(message, 'success', duration);
  };

  /**
   * Show error notification
   * @param {string} message - Error message
   * @param {number} duration - Duration in milliseconds (default: 5000 for errors)
   */
  window.showError = function(message, duration = 5000) {
    return showNotification(message, 'error', duration);
  };

  /**
   * Show warning notification
   * @param {string} message - Warning message
   * @param {number} duration - Duration in milliseconds
   */
  window.showWarning = function(message, duration) {
    return showNotification(message, 'warning', duration);
  };

  /**
   * Show info notification
   * @param {string} message - Info message
   * @param {number} duration - Duration in milliseconds
   */
  window.showInfo = function(message, duration) {
    return showNotification(message, 'info', duration);
  };

  console.log('✅ PhysioMotion notification system loaded');
})();

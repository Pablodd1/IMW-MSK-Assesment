// Authentication utilities for PhysioMotion

// Check if user is authenticated
function checkAuth() {
    const session = localStorage.getItem('clinician_session') || sessionStorage.getItem('clinician_session');
    if (!session) {
        window.location.href = '/static/login.html';
        return null;
    }
    
    try {
        return JSON.parse(session);
    } catch (e) {
        console.error('Error parsing session:', e);
        window.location.href = '/static/login.html';
        return null;
    }
}

// Get current session without redirect
function getSession() {
    const session = localStorage.getItem('clinician_session') || sessionStorage.getItem('clinician_session');
    if (!session) {
        return null;
    }
    
    try {
        return JSON.parse(session);
    } catch (e) {
        console.error('Error parsing session:', e);
        return null;
    }
}

// Logout function
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        localStorage.removeItem('clinician_session');
        sessionStorage.removeItem('clinician_session');
        window.location.href = '/static/login.html';
    }
}

// Display clinician info in navigation
function displayClinicianInfo(elementId = 'clinicianName') {
    const session = getSession();
    if (session) {
        const nameElement = document.getElementById(elementId);
        if (nameElement) {
            const demoLabel = session.is_demo ? ' <span class="text-xs bg-violet-500 text-white px-2 py-1 rounded ml-2">DEMO</span>' : '';
            nameElement.innerHTML = `<i class="fas fa-user-md mr-2"></i>${session.first_name} ${session.last_name}${session.title ? ', ' + session.title : ''}${demoLabel}`;
        }
    }
}

// Add navigation bar with auth info
function addAuthNavigation() {
    const session = getSession();
    if (!session) return;
    
    const demoLabel = session.is_demo ? ' <span class="text-xs bg-violet-500 text-white px-2 py-1 rounded ml-2">DEMO</span>' : '';
    
    return `
        <div class="flex items-center space-x-3 ml-4 pl-4 border-l border-gray-300">
            <span class="text-gray-700 font-medium">
                <i class="fas fa-user-md mr-2"></i>${session.first_name} ${session.last_name}${session.title ? ', ' + session.title : ''}${demoLabel}
            </span>
            <button onclick="logout()" class="text-red-600 hover:text-red-700 transition-colors">
                <i class="fas fa-sign-out-alt mr-1"></i>Logout
            </button>
        </div>
    `;
}

// Initialize auth on page load (add to window.onload or DOMContentLoaded)
if (typeof window !== 'undefined') {
    // Auto-check auth if on protected pages
    if (window.location.pathname !== '/static/login.html' && 
        window.location.pathname !== '/static/register.html' &&
        window.location.pathname !== '/login' &&
        window.location.pathname !== '/register') {
        // Don't auto-redirect, let individual pages handle it
    }
}

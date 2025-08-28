# Fingerprint Management System

A comprehensive Flask-based fingerprint enrollment and verification system with advanced matching algorithms and web interface.

## Features

- **Fingerprint Enrollment**: Register users with their fingerprints using multiple finger positions
- **Real-time Verification**: Verify user identity through fingerprint matching
- **Advanced Matching Algorithms**: Multiple comparison methods including ridge orientation, feature matching, and Local Binary Patterns
- **Web Interface**: Clean, responsive web interface for system management
- **Database Integration**: SQLite database for storing fingerprint templates and verification logs
- **Statistics Dashboard**: Track enrollment and verification statistics
- **MFScan API Integration**: Compatible with MFScan fingerprint scanners

## Requirements

### Hardware
- MFScan compatible fingerprint scanner
- Computer with USB port for scanner connection

### Software Dependencies

```bash
pip install flask opencv-python numpy requests sqlite3 hashlib
```

### Required Python Packages

```python
flask==2.3.3
opencv-python==4.8.1.78
numpy==1.24.3
requests==2.31.0
```

## Installation

1. **Clone or download the system files**
   ```bash
   git clone <repository-url>
   cd mantra-MFS100-python
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up MFScan service**
   - Install MFScan software on your system
   - Ensure the service is running on `https://localhost:8034/mfscan/`
   - Connect your fingerprint scanner

4. **Create required directories**
   ```bash
   mkdir templates static
   ```

## Configuration

### Basic Setup

1. **Update secret key** (Important for production):
   ```python
   app.secret_key = 'your-secure-secret-key-here'
   ```

2. **Configure MFScan URL** (if different):
   ```python
   manager = FingerprintManager(base_url="https://your-mfscan-url/mfscan/")
   ```

3. **Adjust quality thresholds** (optional):
   ```python
   self.MIN_IMAGE_QUALITY = 30
   self.MIN_CONTRAST_THRESHOLD = 20
   self.MIN_SHARPNESS_THRESHOLD = 100
   ```

## Usage

### Starting the System

```bash
python app.py
```

The system will be available at `http://localhost:5000`

### Web Interface Routes

- **`/`** - Main dashboard and verification
- **`/enroll`** - User enrollment page
- **`/verify`** - Standalone verification page
- **`/users`** - View and manage enrolled users
- **`/statistics`** - System statistics and logs
- **`/test_connection`** - Test scanner connectivity

### API Endpoints

#### Test Connection
```http
GET /test_connection
```
Returns scanner connection status and information.

#### Enroll User
```http
POST /enroll
Content-Type: application/x-www-form-urlencoded

user_id=john_doe&finger_position=right_thumb
```

#### Verify User
```http
POST /verify
```
Captures fingerprint and returns verification result.

## Database Schema

### Fingerprints Table
```sql
CREATE TABLE fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    finger_position TEXT NOT NULL,
    fingerprint_hash TEXT NOT NULL,
    bitmap_data TEXT NOT NULL,
    quality INTEGER,
    features TEXT,
    template_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, finger_position)
);
```

### Verification Log Table
```sql
CREATE TABLE verification_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    verification_time TEXT,
    status TEXT,
    similarity_score REAL,
    finger_position TEXT
);
```

## Matching Algorithm

The system uses a multi-modal approach combining:

1. **Ridge Orientation Analysis** (45% weight)
   - Analyzes fingerprint ridge patterns
   - Compares orientation histograms

2. **Advanced Feature Matching** (30% weight)
   - ORB keypoint detection
   - Geometric verification with RANSAC

3. **Local Binary Patterns** (15% weight)
   - Texture analysis comparison
   - Histogram correlation

4. **Template Matching** (10% weight)
   - Normalized cross-correlation
   - Direct image comparison

### Matching Threshold
- Default threshold: **0.55**
- Adjustable based on security requirements
- Higher values = more secure, lower false positives

## File Structure

```
fingerprint-management-system/
├── app.py                 # Main Flask application
├── fingerprints.db        # SQLite database (auto-created)
├── templates/            # HTML templates directory
├── static/               # CSS/JS/images directory
├── fingerprint_system.log # Application logs
└── README.md             # This file
```

## Troubleshooting

### Common Issues

1. **Scanner Not Detected**
   - Verify MFScan service is running
   - Check scanner USB connection
   - Test with: `GET /test_connection`

2. **Low Quality Captures**
   - Clean scanner surface
   - Ensure proper finger placement
   - Adjust quality thresholds if needed

3. **Verification Failures**
   - Check matching threshold settings
   - Verify enrolled fingerprints quality
   - Review system logs

### Error Codes

- **ErrorCode: 0** - Success
- **ErrorCode: 1** - Timeout
- **ErrorCode: 2** - Scanner not found
- **ErrorCode: 3** - Poor quality capture

## Security Considerations

### Production Deployment

1. **Change default secret key**
2. **Use HTTPS in production**
3. **Implement user authentication**
4. **Regular database backups**
5. **Monitor verification logs**

### Data Protection

- Fingerprint data is stored as base64-encoded templates
- Hash-based duplicate detection
- Audit trail for all operations

## Performance Optimization

### Recommended Settings

```python
# For high-security applications
threshold = 0.65
MIN_IMAGE_QUALITY = 50

# For balanced performance
threshold = 0.55
MIN_IMAGE_QUALITY = 30

# For high-throughput scenarios
threshold = 0.45
MIN_IMAGE_QUALITY = 20
```

## API Integration

### MFScan API Methods Used

- **`info`** - Scanner information
- **`capture`** - Fingerprint capture with quality settings

### Request Format
```json
{
    "Quality": 85,
    "TimeOut": 15
}
```

## Logging

The system logs important events including:
- Enrollment attempts
- Verification results
- System errors
- Scanner communication

Logs are stored in `fingerprint_system.log`

## Support

### System Requirements
- Python 3.7+
- Windows/Linux/macOS
- Minimum 2GB RAM
- 100MB free disk space

### Browser Compatibility
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## License

This system is provided as-is for educational and commercial use. Ensure compliance with local biometric data regulations.

## Version History

- **v1.0** - Initial release with basic enrollment/verification
- **v1.1** - Added advanced matching algorithms
- **v1.2** - Enhanced web interface and statistics
- **v1.3** - Improved error handling and logging

# SkillSwap — Peer-to-Peer Skill Exchange Platform

> **Hackathon Submission:** Real-time peer-to-peer knowledge sharing and live video mentorship platform with automated skill matching and WebRTC signaling.

---

## 🌟 Overview
**SkillSwap** connects individuals who want to learn a skill with mentors who possess that skill. Instead of monetary transactions, the platform facilitates a barter-economy for knowledge. Once matched based on their `known` and `learn` profile tags, users can instantly jump into a live WebRTC audio/video call to collaborate and teach in real time.

---

## 🛠️ Key Capabilities & Features
1. **Authenticated User Profiles & Onboarding:** Secure session management with password hashing (`pbkdf2:sha256`) and Flask-Login.
2. **Multi-Skill Tagging System (`/mskills`):** Users tag competencies as `known` (skills they can teach) or `learn` (skills they wish to master).
3. **Automated Bidirectional Matchmaking (`/home`):** Database join queries pair complementary users (e.g. User A wants Python & knows Design; User B knows Python & wants Design).
4. **Zero-Latency WebRTC Video Calling (`/videocall`):**
   - Socket.IO signaling server for room orchestration.
   - P2P direct media streaming with STUN/ICE traversal.
   - Live media track management (camera toggle, microphone mute).

---

## 🏗️ Architecture & Technology Stack
- **Backend:** Python 3, Flask, Flask-SQLAlchemy, Flask-Login, Flask-SocketIO
- **Database:** SQLite (`skillswap.db` with relational User & Skill tables)
- **Real-Time / Signaling:** WebSockets / Socket.IO
- **Peer-to-Peer Media:** WebRTC (`RTCPeerConnection`, `navigator.mediaDevices`)
- **Frontend:** Jinja2 Templates, HTML5/CSS3, JavaScript

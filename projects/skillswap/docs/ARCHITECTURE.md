# SkillSwap — Architecture & Technical Specification

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLIENT BROWSERS                        │
│    User A (Initiator / Learner)     User B (Peer / Mentor)   │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               │ HTTP / Session Auth           │ HTTP / Session Auth
               ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   FLASK APPLICATION SERVER                  │
│  ├── Auth & Routing (Flask-Login, pbkdf2 password hashing)  │
│  ├── Skill Match Engine (SQLAlchemy Relational Joins)       │
│  └── Signaling Gateway (Flask-SocketIO Rooms)               │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
       [SQLite Database]             [Socket.IO Signaling]
     Users, Skills, Categories     Offer / Answer / ICE Candidates
                                               │
                                 ┌─────────────┴─────────────┐
                                 │                           │
                                 ▼                           ▼
                     Peer A <═════════════════════> Peer B
                          Direct WebRTC P2P Media Stream
```

## 2. Relational Data Models
- **`User`**: `id` (Integer PK), `name` (String), `email` (String unique), `phone` (String), `password` (pbkdf2:sha256 hash)
- **`Skill`**: `id` (Integer PK), `user_id` (Integer FK -> `User.id`), `skill_name` (String), `category` (`'known'` | `'learn'`)

## 3. Matchmaking Algorithm
When a user accesses their personalized dashboard (`/home`):
1. The engine queries up to 3 target skills marked as `category == 'learn'` for `current_user`.
2. For each desired skill, a join query executes between `User` and `Skill`:
   ```python
   db.session.query(User)\
       .join(Skill, User.id == Skill.user_id)\
       .filter(Skill.skill_name == skill.skill_name,
               Skill.category == 'known',
               User.id != user.id)\
       .first()
   ```
3. Compatible mentors are paired and presented with a 1-click call invite linking to a shared room.

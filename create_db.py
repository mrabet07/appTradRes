from flask_auth.app import db, app

with app.app_context():
    db.create_all()

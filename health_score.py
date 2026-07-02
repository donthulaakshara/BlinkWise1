def calculate_health_score(blink_rate, session_minutes, alert_count):

    # Blink Rate Score (50 Points)

    if blink_rate >= 15:
        blink_score = 50

    elif blink_rate >= 12:
        blink_score = 40

    elif blink_rate >= 10:
        blink_score = 30

    elif blink_rate >= 8:
        blink_score = 20

    else:
        blink_score = 10

    # Alert Score (30 Points)

    alert_score = max(
        30 - (alert_count * 5),
        0
    )

    # Session Score (20 Points)

    if session_minutes < 30:
        session_score = 20

    elif session_minutes < 60:
        session_score = 15

    elif session_minutes < 120:
        session_score = 10

    else:
        session_score = 5

    # Final Score

    total_score = (
        blink_score
        + alert_score
        + session_score
    )

    return total_score
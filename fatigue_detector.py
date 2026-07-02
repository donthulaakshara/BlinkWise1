def detect_fatigue(health_score):

    if health_score >= 90:
        return "EXCELLENT", (80, 255, 120)

    elif health_score >= 75:
        return "GOOD", (80, 255, 120)

    elif health_score >= 60:
        return "EYE STRAIN", (255, 220, 0)

    elif health_score >= 40:
        return "FATIGUE", (255, 165, 0)

    else:
        return "TAKE A BREAK", (255, 80, 80)
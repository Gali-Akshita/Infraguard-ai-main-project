def risk_level(confidence):

    if confidence > 0.85:
        return "HIGH", 30, "Immediate inspection required"

    elif confidence > 0.65:
        return "MEDIUM", 60, "Schedule maintenance"

    else:
        return "LOW", 90, "Structure appears stable"

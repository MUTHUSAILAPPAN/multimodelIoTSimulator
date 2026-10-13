"""
Scenario execution engine for Multimodal IoT house simulator.
Translates user object actions into multimodal sensor signals, audio clips,
image features, and visual state overlays.
"""
from room_config import ACTIONS_REGISTRY

def get_scenario_for_action(action_id, object_name, current_view):
    """
    Returns the simulation state payload corresponding to a specific action and object.
    """
    base_state = {
        "action_id": action_id,
        "object_name": object_name,
        "view": current_view,
        "active": True,
        "hold_prev": True,
        "sfx_file": f"{action_id}.wav" if action_id else None,
    }

    if action_id == "simulate_glass_break":
        return {
            **base_state,
            "scenario_name": f"Glass Break ({object_name})",
            "vib": 86,
            "motion": 20,
            "temp": 24,
            "aud_mode": "Sample clip",
            "aud_sample": "Glass break",
            "img_mode": "Simulated",
            "fire": 0.0,
            "person": False,
            "visual_overlay": {
                "type": "crack",
                "object": object_name,
                "label": "STRUCTURAL FRACTURE DETECTED",
            },
        }

    elif action_id in ("simulate_forced_entry", "inspect_intruder"):
        return {
            **base_state,
            "scenario_name": f"Intrusion / Forced Entry ({object_name})",
            "motion": 85,
            "prev": 10,
            "vib": 25,
            "temp": 24,
            "aud_mode": "Sample clip",
            "aud_sample": "Heavy thud",
            "img_mode": "Sample scene",
            "img_sample": "Intruder scene",
            "fire": 0.0,
            "person": True,
            "visual_overlay": {
                "type": "intruder",
                "object": object_name,
                "label": "AI VISION: INTRUSION DETECTED [96.8% CONF]",
            },
        }

    elif action_id == "simulate_tv_fire":
        return {
            **base_state,
            "scenario_name": f"Fire Incident ({object_name})",
            "temp": 82,
            "motion": 25,
            "vib": 10,
            "aud_mode": "Sample clip",
            "aud_sample": "Loud noise burst",
            "img_mode": "Sample scene",
            "img_sample": "Fire scene",
            "fire": 0.38,
            "person": False,
            "visual_overlay": {
                "type": "fire",
                "object": object_name,
                "label": "THERMAL IGNITION & FLAME DETECTED",
            },
        }

    elif action_id in ("simulate_tv_fault", "simulate_fan_fault"):
        return {
            **base_state,
            "scenario_name": f"Equipment Fault ({object_name})",
            "vib": 88,
            "motion": 15,
            "temp": 34,
            "aud_mode": "Sample clip",
            "aud_sample": "Machine hum",
            "img_mode": "Simulated",
            "fire": 0.0,
            "person": False,
            "visual_overlay": {
                "type": "warning_gear",
                "object": object_name,
                "label": "BEARING VIBRATION / HARMONIC HUM",
            },
        }

    elif action_id == "simulate_extreme_temp":
        return {
            **base_state,
            "scenario_name": f"HVAC Overheat ({object_name})",
            "temp": 78,
            "vib": 12,
            "motion": 10,
            "aud_mode": "Sample clip",
            "aud_sample": "Quiet room",
            "img_mode": "Simulated",
            "fire": 0.0,
            "person": False,
            "visual_overlay": {
                "type": "thermal_hot",
                "object": object_name,
                "label": "TEMPERATURE EXCEEDED (78 °C > 70 °C LIMIT)",
            },
        }

    elif action_id == "simulate_freeze_temp":
        return {
            **base_state,
            "scenario_name": f"HVAC Freeze Condition ({object_name})",
            "temp": -6,
            "vib": 8,
            "motion": 10,
            "aud_mode": "Sample clip",
            "aud_sample": "Quiet room",
            "img_mode": "Simulated",
            "fire": 0.0,
            "person": False,
            "visual_overlay": {
                "type": "thermal_cold",
                "object": object_name,
                "label": "SUB-ZERO FREEZE DETECTED (-6 °C < 5 °C LIMIT)",
            },
        }

    elif action_id in ("simulate_table_impact", "simulate_sofa_fall", "simulate_object_fall"):
        return {
            **base_state,
            "scenario_name": f"Sudden Fall / Impact ({object_name})",
            "motion": 92,
            "prev": 6,  # results in delta = 86
            "vib": 40,
            "temp": 24,
            "aud_mode": "Sample clip",
            "aud_sample": "Heavy thud",
            "img_mode": "Simulated",
            "fire": 0.0,
            "person": False,
            "visual_overlay": {
                "type": "impact",
                "object": object_name,
                "label": "IMPACT SHOCKWAVE / MOTION DELTA 86",
            },
        }

    elif action_id == "simulate_loud_knock":
        return {
            **base_state,
            "scenario_name": f"Unauthorized Loud Noise ({object_name})",
            "motion": 72,
            "vib": 25,
            "temp": 24,
            "aud_mode": "Sample clip",
            "aud_sample": "Loud noise burst",
            "img_mode": "Simulated",
            "fire": 0.0,
            "person": False,
            "visual_overlay": {
                "type": "sound_wave",
                "object": object_name,
                "label": "HIGH ACOUSTIC ENERGY DETECTED",
            },
        }

    elif action_id == "simulate_sofa_motion":
        return {
            **base_state,
            "scenario_name": f"Occupant Presence ({object_name})",
            "motion": 65,
            "prev": 60,
            "vib": 10,
            "temp": 24,
            "aud_mode": "Sample clip",
            "aud_sample": "Quiet room",
            "img_mode": "Simulated",
            "fire": 0.0,
            "person": False,
            "visual_overlay": {
                "type": "motion_zone",
                "object": object_name,
                "label": "NORMAL RESIDENTIAL ACTIVITY (NO ALARM)",
            },
        }

    elif action_id == "toggle_lights":
        return {
            **base_state,
            "scenario_name": f"Lighting Toggled ({object_name})",
            "toggle_lights": True,
            "visual_overlay": None,
        }

    return base_state

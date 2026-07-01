import cv2
import numpy as np
import os
import copy
import socket
import pickle


def circular_hue_distance(h1, h2):
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


def classify_color(h, s, v, b, g, r, l, a, lab_b):
    h, s, v = int(h), int(s), int(v)
    b, g, r = int(b), int(g), int(r)
    l, a, lab_b = int(l), int(a), int(lab_b)

    max_c = max(r, g, b)
    min_c = min(r, g, b)
    chroma = max_c - min_c
    rg_ratio = g / max(r, 1)
    lab_chroma = ((a - 128) ** 2 + (lab_b - 128) ** 2) ** 0.5
    r_minus_g = r - g
    g_minus_b = g - b
    a_minus_b = a - lab_b

    # White under webcam tint can shift toward cyan/blue.
    # LAB chroma is a stronger "neutral color" test than HSV alone.
    if (lab_chroma <= 24 and l >= 115) or (lab_chroma <= 32 and v >= 135):
        return "W"
    if (s <= 95 and v >= 90) or (chroma <= 45 and v >= 80):
        return "W"
    # Cyan-looking white from webcam AWB tends to land here; catch it before blue.
    if 88 <= h <= 120 and v >= 120 and lab_chroma <= 36:
        return "W"

    # Yellow, green, blue are relatively stable in HSV.
    if 18 <= h <= 42 and s >= 70 and v >= 70:
        return "Y"
    if 38 <= h <= 95 and s >= 55 and v >= 45:
        return "G"
    if 90 <= h <= 145 and s >= 50:
        return "B"

    # Red vs orange:
    # For this webcam, LAB a-b separates them better than hue alone.
    if h >= 170 or h <= 30:
        if h >= 170:
            return "R"
        if h <= 8:
            # Very low hue is usually red unless LAB says strongly yellowish-warm.
            if a_minus_b >= 8 or r_minus_g >= 12:
                return "R"
            return "O"
        if 9 <= h <= 18:
            # Mid warm band: use LAB + RGB jointly.
            if a_minus_b >= 18 and r_minus_g >= 8:
                return "R"
            if a_minus_b <= 8 or g_minus_b >= 6 or rg_ratio >= 0.90:
                return "O"
            return "O"
        # 19..30 tends to orange; keep a narrow red rescue for shifted cameras.
        if a_minus_b >= 28 and r_minus_g >= 20:
            return "R"
        return "O"

    # Fallback by nearest hue prototype.
    prototypes = {
        "R": 0,
        "O": 16,
        "Y": 30,
        "G": 60,
        "B": 110,
    }
    return min(prototypes, key=lambda c: circular_hue_distance(h, prototypes[c]))


def sample_patch_stats(bgr_image, hsv_image, lab_image, x, y, radius=10):
    h_img, w_img = hsv_image.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(w_img, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h_img, y + radius + 1)
    patch_hsv = hsv_image[y0:y1, x0:x1].reshape(-1, 3)
    patch_bgr = bgr_image[y0:y1, x0:x1].reshape(-1, 3)
    patch_lab = lab_image[y0:y1, x0:x1].reshape(-1, 3)
    h, s, v = np.median(patch_hsv, axis=0)
    b, g, r = np.median(patch_bgr, axis=0)
    l, a, lab_b = np.median(patch_lab, axis=0)
    return int(h), int(s), int(v), int(b), int(g), int(r), int(l), int(a), int(lab_b)

def get_position_for_move(move, frame_size, image_size):
    frame_h, frame_w = frame_size
    if move in ["R", "R'"]:
        return (520, 195)
    elif move in ["L", "L'"]:
        return (200, 195)
    elif move in ["U", "U'"]:
        return (260, 145)
    elif move in ["D", "D'"]:
        return (260, 465)
    else:
        return (250, 240)

def overlay_image(bg, overlay, position):
    h, w = overlay.shape[:2]
    x, y = position
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (1 - alpha) * bg[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    else:
        bg[y:y+h, x:x+w] = overlay
    return bg

def draw_arrow_for_move(frame, move):
    image_path = f"resources/{move}.png"
    h, w = frame.shape[:2]
    size = (150, 150)
    if os.path.exists(image_path):
        overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if overlay is not None:
            position = get_position_for_move(move, (h, w), size)
            frame[:] = overlay_image(frame, overlay, position)
    cv2.putText(frame, f"Move: {move}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


def expand_moves(solution_str):
    expanded = []
    for move in solution_str.split():
        if move == "B":
            expanded.extend(["TURN_BACK", "F", "TURN_BACK"])
        elif move == "B'":
            expanded.extend(["TURN_BACK", "F'", "TURN_BACK"])
        elif move == "B2":
            expanded.extend(["TURN_BACK", "F", "F", "TURN_BACK"])
        elif move.endswith("2"):
            expanded.extend([move[0], move[0]])
        else:
            expanded.append(move)
    return expanded

def get_required_presses(move):
    if move.endswith("2"):
        return 2 if move[0] != 'B' else 4
    elif move[0] == 'B':
        return 3
    else:
        return 1

def rotate_face(face, turns=1):
    for _ in range(turns % 4):
        face[:] = [
            face[6], face[3], face[0],
            face[7], face[4], face[1],
            face[8], face[5], face[2]
        ]
    return face

def cycle_edges(state, faces, indices, turns=1):
    for _ in range(turns % 4):
        tmp = [state[faces[-1]][i] for i in indices[-1]]
        for i in reversed(range(1, 4)):
            for j in range(3):
                state[faces[i]][indices[i][j]] = state[faces[i - 1]][indices[i - 1][j]]
        for j in range(3):
            state[faces[0]][indices[0][j]] = tmp[j]

def apply_move(state, move):
    face = move[0]
    modifier = move[1:] if len(move) > 1 else ''
    turns = {'': 1, "'": 3, '2': 2}[modifier]
    state = copy.deepcopy(state)
    rotate_face(state[face], turns)
    if face == 'U':
        cycle_edges(state, ['B', 'R', 'F', 'L'], [[0,1,2]]*4, turns)
    elif face == 'D':
        cycle_edges(state, ['F', 'R', 'B', 'L'], [[6,7,8]]*4, turns)
    elif face == 'F':
        cycle_edges(state, ['U', 'R', 'D', 'L'], [[6,7,8], [0,3,6], [2,1,0], [8,5,2]], turns)
    elif face == 'B':
        cycle_edges(state, ['U', 'L', 'D', 'R'], [[2,1,0], [0,3,6], [6,7,8], [8,5,2]], turns)
    elif face == 'L':
        cycle_edges(state, ['U', 'F', 'D', 'B'], [[0,3,6]]*3 + [[8,5,2]], turns)
    elif face == 'R':
        cycle_edges(state, ['U', 'B', 'D', 'F'], [[8,5,2], [0,3,6], [8,5,2], [8,5,2]], turns)
    return state

def print_cube(state):
    for face in ['U', 'R', 'F', 'D', 'L', 'B']:
        print(f"{face}: {state[face]}")


VALID_COLORS = {'W', 'R', 'O', 'Y', 'G', 'B'}


def parse_face_pos(face, pos_text, face_order):
    face = face.upper()
    if face not in face_order:
        raise ValueError(f"Face must be one of {face_order}.")
    pos = int(pos_text)
    if pos < 1 or pos > 9:
        raise ValueError("Position must be 1..9.")
    return face, pos - 1


def show_scanned_faces(cube_faces, face_order):
    print("\nScanned faces (index:color):")
    for face in face_order:
        stickers = cube_faces.get(face)
        if stickers is None:
            continue
        print(f"{face} (center={stickers[4]})")
        for r in range(3):
            base = r * 3
            row = [f"{base + c + 1}:{stickers[base + c]}" for c in range(3)]
            print("  " + " ".join(row))


def get_color_counts(cube_faces, face_order):
    counts = {c: 0 for c in sorted(VALID_COLORS)}
    unknown = 0
    for face in face_order:
        for sticker in cube_faces.get(face, []):
            if sticker in counts:
                counts[sticker] += 1
            else:
                unknown += 1
    return counts, unknown


def build_cube_string(cube_faces, face_order):
    color_to_face = {}
    for face in face_order:
        center = cube_faces[face][4]
        if center in color_to_face:
            other = color_to_face[center]
            raise ValueError(f"Duplicate center color {center}: {other} and {face}.")
        color_to_face[center] = face

    cube_string = ''.join(
        color_to_face.get(sticker, '?')
        for face in face_order
        for sticker in cube_faces[face]
    )
    return cube_string, color_to_face


def build_cube_string_from_center_prototypes(cube_face_stats, face_order):
    # Use per-session center stickers as LAB prototypes, then rebalance to 9 stickers/face.
    if any(face not in cube_face_stats for face in face_order):
        raise ValueError("Missing scanned face stats.")

    prototypes = []
    for face in face_order:
        stats = cube_face_stats[face]
        if len(stats) != 9:
            raise ValueError(f"Face {face} does not have 9 sampled stickers.")
        l, a, lab_b = stats[4][:3]
        prototypes.append(np.array([l, a, lab_b], dtype=np.float32))

    sticker_features = []
    for face in face_order:
        for idx in range(9):
            l, a, lab_b = cube_face_stats[face][idx][:3]
            sticker_features.append(np.array([l, a, lab_b], dtype=np.float32))

    num_faces = len(face_order)
    num_stickers = len(sticker_features)
    target = num_stickers // num_faces

    distances = []
    for feat in sticker_features:
        d = [float(np.linalg.norm(feat - proto)) for proto in prototypes]
        distances.append(d)

    assign = [int(np.argmin(d)) for d in distances]
    counts = [0] * num_faces
    for a in assign:
        counts[a] += 1

    # Rebalance to exactly 9 stickers per face by minimum-penalty moves.
    guard = 0
    while True:
        over = [i for i, c in enumerate(counts) if c > target]
        under = [i for i, c in enumerate(counts) if c < target]
        if not over and not under:
            break
        best = None  # (penalty, sticker_idx, from_face, to_face)
        for s_idx, cur_face in enumerate(assign):
            if counts[cur_face] <= target:
                continue
            for dst_face in under:
                penalty = distances[s_idx][dst_face] - distances[s_idx][cur_face]
                if best is None or penalty < best[0]:
                    best = (penalty, s_idx, cur_face, dst_face)
        if best is None:
            raise ValueError("Unable to rebalance prototype assignment.")
        _, s_idx, src_face, dst_face = best
        assign[s_idx] = dst_face
        counts[src_face] -= 1
        counts[dst_face] += 1
        guard += 1
        if guard > 2000:
            raise ValueError("Prototype rebalance exceeded iteration guard.")

    if any(c != target for c in counts):
        raise ValueError(f"Prototype assignment imbalance: {counts}")

    cube_string = ''.join(face_order[a] for a in assign)
    return cube_string, counts


def mirror_face_lr(face):
    # Horizontal mirror on a 3x3 face.
    return [face[2], face[1], face[0], face[5], face[4], face[3], face[8], face[7], face[6]]


def mirror_all_faces_lr(cube_faces, face_order):
    return {face: mirror_face_lr(cube_faces[face]) for face in face_order}


def correction_shell(cube_faces, face_order):
    print("\nManual correction mode (optional).")
    print("Commands:")
    print("  show")
    print("  set <FACE> <POS 1-9> <COLOR W/R/O/Y/G/B>")
    print("  swap <FACE1> <POS1> <FACE2> <POS2>")
    print("  string")
    print("  setstring <54 chars using U,R,F,D,L,B>")
    print("  done")

    override_cube_string = None
    while True:
        counts, unknown = get_color_counts(cube_faces, face_order)
        counts_text = " ".join(f"{k}={v}" for k, v in counts.items())
        print(f"\nCounts: {counts_text} Unknown={unknown}")
        try:
            candidate_cube_string, _ = build_cube_string(cube_faces, face_order)
            print(f"Current cube string: {candidate_cube_string}")
        except Exception as e:
            print(f"Current cube string: <invalid> ({e})")

        raw = input("edit> ").strip()
        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "done":
            return override_cube_string
        if cmd == "show":
            show_scanned_faces(cube_faces, face_order)
            continue
        if cmd == "string":
            try:
                s, _ = build_cube_string(cube_faces, face_order)
                print(s)
            except Exception as e:
                print(f"Cannot build string: {e}")
            continue
        if cmd == "setstring":
            if len(parts) != 2:
                print("Usage: setstring <54 chars using U,R,F,D,L,B>")
                continue
            s = parts[1].strip().upper()
            if len(s) != 54 or any(ch not in "URFDLB" for ch in s):
                print("Invalid cube string. Need exactly 54 chars from U,R,F,D,L,B.")
                continue
            override_cube_string = s
            print("Custom cube string set. Type 'done' to solve with it.")
            continue
        if cmd == "set":
            if len(parts) != 4:
                print("Usage: set <FACE> <POS> <COLOR>")
                continue
            try:
                face, idx = parse_face_pos(parts[1], parts[2], face_order)
                color = parts[3].upper()
                if color not in VALID_COLORS:
                    raise ValueError("Color must be one of W,R,O,Y,G,B.")
                cube_faces[face][idx] = color
                override_cube_string = None
                print(f"Updated {face}{idx + 1} -> {color}")
            except Exception as e:
                print(f"Set failed: {e}")
            continue
        if cmd == "swap":
            if len(parts) != 5:
                print("Usage: swap <FACE1> <POS1> <FACE2> <POS2>")
                continue
            try:
                f1, i1 = parse_face_pos(parts[1], parts[2], face_order)
                f2, i2 = parse_face_pos(parts[3], parts[4], face_order)
                cube_faces[f1][i1], cube_faces[f2][i2] = cube_faces[f2][i2], cube_faces[f1][i1]
                override_cube_string = None
                print(f"Swapped {f1}{i1 + 1} <-> {f2}{i2 + 1}")
            except Exception as e:
                print(f"Swap failed: {e}")
            continue

        print("Unknown command. Use: show, set, swap, string, setstring, done")


cap = cv2.VideoCapture(0)

GRID_SIZE = 3
SPACING = 160
DOT_RADIUS = 5
face_order = ['U', 'R', 'F', 'D', 'L', 'B']
cube_faces = {}
cube_face_stats = {}

print("▶️ Press keys: u r f d l b to scan that face")
print("▶️ Press ESC when done")
print("▶️ Live sticker letters are preview only; final solve uses center-prototype matching.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if ret:
        frame = cv2.resize(frame, (750, 640))
    height, width = 640, 750
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    center_x, center_y = width // 2, height // 2

    current_face = []
    current_face_stats = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = center_x + (j - 1) * SPACING
            y = center_y + (i - 1) * SPACING + 50
            h, s, v, b, g, r, l, a, lab_b = sample_patch_stats(frame, hsv, lab, x, y, radius=10)
            color = classify_color(h, s, v, b, g, r, l, a, lab_b)
            current_face.append(color)
            current_face_stats.append((l, a, lab_b, h, s, v, b, g, r))
            cv2.circle(frame, (x, y), DOT_RADIUS, (0, 255, 0), -1)
            cv2.putText(frame, color, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Cube Scanner", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif chr(key).upper() in face_order:
        face_key = chr(key).upper()
        cube_faces[face_key] = current_face.copy()
        cube_face_stats[face_key] = current_face_stats.copy()
        print(f"✅ Scanned {face_key}:")
        for i in range(0, 9, 3):
            print(current_face[i], current_face[i + 1], current_face[i + 2])

cap.release()
cv2.destroyAllWindows()

if len(cube_faces) == 6:
    show_scanned_faces(cube_faces, face_order)
    prototype_cube_string = None
    if len(cube_face_stats) == 6:
        try:
            prototype_cube_string, proto_counts = build_cube_string_from_center_prototypes(cube_face_stats, face_order)
            print("\nPrototype-based cube string (center-distance balanced):")
            print(prototype_cube_string)
            print(f"Prototype counts by face order {face_order}: {proto_counts}")
        except Exception as e:
            print(f"\nPrototype mapping unavailable: {e}")

    cube_faces_before_edit = copy.deepcopy(cube_faces)
    override_cube_string = correction_shell(cube_faces, face_order)
    faces_changed = (cube_faces != cube_faces_before_edit)
    print("\nBuilding cube string for solver...")
    try:
        auto_cube_string, _ = build_cube_string(cube_faces, face_order)
    except Exception as e:
        auto_cube_string = None
        print(f"Automatic string build failed: {e}")

    if faces_changed and prototype_cube_string:
        print("Manual sticker edits detected; skipping prototype cube string.")
        prototype_cube_string = None

    cube_string = override_cube_string or prototype_cube_string or auto_cube_string
    if not cube_string:
        print("No valid cube string available. Re-run scan and correct entries.")
        raise SystemExit(1)

    print("\nFinal cube string:")
    print(cube_string)
    if override_cube_string:
        print("Using manual string override from correction mode.")
    elif prototype_cube_string:
        print("Using prototype-based cube string (robust against color confusion).")

    try:
        import kociemba
        cube_faces_for_solution = {face: cube_faces[face][:] for face in face_order}
        solve_cube_string = cube_string
        try:
            solution = kociemba.solve(solve_cube_string)
        except Exception as solve_err:
            # If webcam feed is mirrored, try mirrored face stickers automatically.
            mirrored_faces = mirror_all_faces_lr(cube_faces, face_order)
            mirrored_cube_string, _ = build_cube_string(mirrored_faces, face_order)
            print("\nPrimary solve failed; trying mirrored-entry correction...")
            try:
                solution = kociemba.solve(mirrored_cube_string)
                cube_faces_for_solution = mirrored_faces
                solve_cube_string = mirrored_cube_string
                print("Mirrored-entry correction worked.")
                print("Mirrored cube string:")
                print(solve_cube_string)
            except Exception:
                raise solve_err
        print("\n🧩 Solution:")
        print(solution)

        kociemba_moves = solution.strip().split()
        overlay_moves = expand_moves(solution)
        cube_state = {face: cube_faces_for_solution[face][:] for face in face_order}
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', 9999))
            viewer_connected = True
            sock.send(pickle.dumps(cube_state))
        except ConnectionRefusedError:
            print("⚠️ Viewer not running. Continuing without visual updates.")
            viewer_connected = False

        if viewer_connected:
            try:
                sock.send(pickle.dumps(cube_state))
            except Exception as e:
                print("⚠️ Failed to send cube state to viewer:", e)

        cap = cv2.VideoCapture(0)

        current_overlay_step = 0
        logical_step = 0
        presses_remaining = get_required_presses(kociemba_moves[logical_step]) if kociemba_moves else 0

        while current_overlay_step < len(overlay_moves):
            is_ok, frame = cap.read()
            if not is_ok:
                break
            if is_ok:
                frame = cv2.resize(frame, (750, 640))
            overlay_move = overlay_moves[current_overlay_step]
            if overlay_move != "TURN_BACK":
                draw_arrow_for_move(frame, overlay_move)
            else:
                if overlay_move != "TURN_BACK":
                    draw_arrow_for_move(frame, overlay_move)
                else:
                    cv2.putText(frame, "Rotate cube to back", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    image_path = "resources/TURN_BACK.png"
                    turn_back_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if turn_back_img is not None:
                        h, w = frame.shape[:2]
                        x = (w - turn_back_img.shape[1]) // 2
                        y = (h - turn_back_img.shape[0]) // 2
                        frame = overlay_image(frame, turn_back_img, (x, y))

            cv2.imshow("Cube Solver", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                move = kociemba_moves[logical_step]
                print(f"🔁 Step {logical_step+1}: Move {move} - Presses left: {presses_remaining - 1}")
                presses_remaining -= 1
                current_overlay_step += 1

                if presses_remaining == 0:
                    cube_state = apply_move(cube_state, move)
                    print(f"✅ Move {move} completed and applied.")
                    sock.send(pickle.dumps(cube_state))
                    print_cube(cube_state)
                    logical_step += 1
                    if logical_step < len(kociemba_moves):
                        presses_remaining = get_required_presses(kociemba_moves[logical_step])

            if key == 27:
                break

        print("🎉 Cube solved! Showing final state. Press ESC to exit.")
        while True:
            is_ok, frame = cap.read()
            if not is_ok:
                break
            frame = cv2.resize(frame, (750, 640))
            cv2.putText(frame, "Cube Solved!", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Cube Solver", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


    except Exception as e:
        print("⚠️ Could not solve:", e)
else:
    print("⚠️ Scan all 6 faces! Scanned faces:", list(cube_faces.keys()))




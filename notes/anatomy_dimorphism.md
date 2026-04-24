# Anatomical sex dimorphism — implementation notes

Sources (general): Farkas anthropometry; Drillis & Contini (1966) body
segment proportions; SMPL/SMPL-X gender PCA; Pheasant "Bodyspace" 1996;
Reynolds & Karlsson Q-angle data.

## Skeletal differences
- **Shoulder breadth (biacromial)**: M ≈ 0.245·H, F ≈ 0.213·H. Ratio ~1.15.
- **Hip width (biiliac)**: M ≈ 0.191·H, F ≈ 0.197·H. Female hips ≈ shoulders.
- **Shoulder-to-hip ratio**: M ≈ 1.18, F ≈ 1.03.
- **Pelvis**: female pelvis is wider, shallower (A-P), greater subpubic angle.
  → wider hip_w on female + slight outward femoral head spacing.
- **Q-angle (hip-to-knee angle)**: M ≈ 11°, F ≈ 17°. Female femurs angle
  inward toward the knees.
- **Sternum/clavicle**: male clavicles longer + flatter; female clavicles
  shorter + more S-curved.
- **Jaw/mandible**: male wider + squarer; female narrower + more pointed.
- **Neck**: male circumference ~38–40 cm, female ~32–34 cm. ~15% thicker.
- **Limb length proportions**: roughly the same per height; male slightly
  longer forearms (brachial index).

## Soft-tissue / fat distribution
- **Body fat %**: M typical 12–20, F typical 22–32. Even at the SAME total
  bulk, the *distribution* differs.
- **Gynoid (female) distribution**: subcutaneous fat concentrates on
  hips, outer thighs, glutes, lower abdomen. Waist stays narrow.
  → WHR (waist:hip) F ≈ 0.70, M ≈ 0.90.
- **Android (male) distribution**: fat goes to abdomen + chest. Waist
  approaches hips.
- **Bust**: chest fat pads of variable size; sit on the pectoral plane;
  teardrop projection (slightly downward) at rest, base spread ~1/3 of
  chest width per side, separation ~1 fingerbreadth at sternum.
- **Glutes**: female gluteal mass larger and rounder; sits posterior +
  inferior to the pelvis bone.
- **Thigh taper**: female thigh is fuller at hip and tapers MORE
  gradually to the knee than male. Male thigh shows a more pronounced
  vastus lateralis bulge mid-thigh.
- **Upper arm**: male shows deltoid + biceps bulges; female arm has
  smoother, more uniform taper with slightly larger triceps fat pad
  mid-upper-arm.
- **Calf**: male gastrocnemius bulge is more pronounced and higher.

## What's already in this repo
- `Shape`: gender, height, bulk, limb_len, shoulder_w, hip_w, bust,
  head_size, leg_len, torso_len. Single global `bulk` scales ALL bone radii.
- Female-only meshes: bust ellipsoids on chest, glutes ellipsoids on pelvis.
- `_limb_profile` is gender-agnostic.
- No waist narrowing, no Q-angle, no neck thickness control, no per-region
  fat distribution.

## What this cycle adds
1. `waist` knob (default 1.0 male, ~0.78–0.88 female) → scales `spine`
   radius and biases the chest bottom-cap inward to suggest a waistline.
2. `lower_bulk` knob → multiplies thigh/glute/pelvis radii independently
   from upper body bulk. Female lower_bulk ≥ upper bulk.
3. `neck_thick` knob → male 1.0–1.10, female 0.78–0.88.
4. Gender-aware `_limb_profile`: female thigh tapers less; male upper-arm
   has a stronger mid-belly bulge.
5. Q-angle: shift the thigh head outward and the thigh tip slightly inward
   for females so the knees track inside the hip line.
6. Bust ellipsoid: redo positioning (lower, wider base, more projection),
   teardrop axes.
7. Glutes: lift slightly and round more on female; tied to lower_bulk.
8. Deltoid bulge: bigger for male, smaller for female (already-existing
   shoulder ellipsoid scaled by gender).

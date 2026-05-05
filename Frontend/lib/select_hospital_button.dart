import 'package:flutter/material.dart';

/// Matches [BhColors.slate] in `main.dart` (`0xFF393E46`).
const _kSlate = Color(0xFF393E46);

/// Primary nav action to choose a hospital (wire [onPressed] when flow exists).
class SelectHospitalButton extends StatelessWidget {
  const SelectHospitalButton({
    super.key,
    this.onPressed,
    this.widthFactor = 0.4,
    this.height = 52,
  });

  final VoidCallback? onPressed;
  final double widthFactor;
  final double height;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width * widthFactor;
    return SizedBox(
      width: width,
      height: height,
      child: FilledButton(
        onPressed: onPressed ?? () {},
        style: FilledButton.styleFrom(
          backgroundColor: _kSlate,
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: const Text('Select hospital'),
      ),
    );
  }
}

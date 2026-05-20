import 'package:flutter/material.dart';

/// Search field for filtering patients (wire [onChanged] when filtering is implemented).
class PatientSearchBar extends StatelessWidget {
  const PatientSearchBar({
    super.key,
    required this.controller,
    this.onChanged,
    this.onFilterTap,
    this.isFilterActive = false,
    this.hintText = 'Search patients',
  });

  final TextEditingController controller;
  final ValueChanged<String>? onChanged;
  final VoidCallback? onFilterTap;
  final bool isFilterActive;
  final String hintText;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.black12),
      ),
      child: TextField(
        controller: controller,
        onChanged: onChanged,
        decoration: InputDecoration(
          prefixIcon: const Icon(Icons.search),
          suffixIcon: Padding(
            padding: const EdgeInsets.only(right: 6),
            child: IconButton(
              onPressed: onFilterTap,
              tooltip: 'Filter patients',
              icon: Icon(
                Icons.tune_rounded,
                color: isFilterActive ? Theme.of(context).colorScheme.primary : Colors.black54,
              ),
            ),
          ),
          hintText: hintText,
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 16,
          ),
        ),
      ),
    );
  }
}

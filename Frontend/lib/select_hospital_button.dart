import 'package:flutter/material.dart';

import 'demo_hospitals.dart';

/// Matches [BhColors.slate] in `main.dart` (`0xFF393E46`).
const _kSlate = Color(0xFF393E46);

/// Primary nav action to choose a hospital.
class SelectHospitalButton extends StatelessWidget {
  const SelectHospitalButton({
    super.key,
    required this.selectedHospital,
    required this.onHospitalSelected,
    this.widthFactor = 0.4,
    this.height = 52,
  });

  final DemoHospital? selectedHospital;
  final ValueChanged<DemoHospital> onHospitalSelected;
  final double widthFactor;
  final double height;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: height,
      height: height,
      child: FilledButton(
        onPressed: () => _openHospitalPicker(context),
        style: FilledButton.styleFrom(
          backgroundColor: _kSlate,
          foregroundColor: Colors.white,
          padding: EdgeInsets.zero,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: Text(
          selectedHospital == null ? 'Select hospital' : 'Hospital: ${selectedHospital!.city}',
          overflow: TextOverflow.ellipsis,
        ),
      ),
    );
  }

  Future<void> _openHospitalPicker(BuildContext context) async {
    final picked = await showDialog<DemoHospital>(
      context: context,
      builder: (context) => _HospitalPickerDialog(
        initialSelection: selectedHospital,
        hospitals: demoHospitals,
      ),
    );

    if (picked != null) {
      onHospitalSelected(picked);
    }
  }
}

class _HospitalPickerDialog extends StatefulWidget {
  const _HospitalPickerDialog({
    required this.hospitals,
    required this.initialSelection,
  });

  final List<DemoHospital> hospitals;
  final DemoHospital? initialSelection;

  @override
  State<_HospitalPickerDialog> createState() => _HospitalPickerDialogState();
}

class _HospitalPickerDialogState extends State<_HospitalPickerDialog> {
  final TextEditingController _queryController = TextEditingController();

  String _query = '';
  late DemoHospital? _selected = widget.initialSelection;

  List<DemoHospital> get _filteredHospitals {
    final q = _query.trim().toLowerCase();
    if (q.isEmpty) {
      return widget.hospitals;
    }
    return widget.hospitals.where((h) {
      final haystack = '${h.zip} ${h.addressLine1} ${h.city} ${h.state} ${h.name}'.toLowerCase();
      return haystack.contains(q);
    }).toList();
  }

  @override
  void dispose() {
    _queryController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return AlertDialog(
      title: const Text('Select a hospital (demo)'),
      content: SizedBox(
        width: 520,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _queryController,
              onChanged: (v) => setState(() => _query = v),
              decoration: const InputDecoration(
                prefixIcon: Icon(Icons.search),
                labelText: 'Search by zip code or address',
                hintText: 'e.g. 33176, Kendall, Jacksonville, 32207...',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                'Locations',
                style: theme.textTheme.labelLarge?.copyWith(fontWeight: FontWeight.w700),
              ),
            ),
            const SizedBox(height: 8),
            Flexible(
              child: Material(
                color: Colors.transparent,
                child: ListView.separated(
                  shrinkWrap: true,
                  itemCount: _filteredHospitals.length,
                  separatorBuilder: (context, _) => const Divider(height: 1),
                  itemBuilder: (context, index) {
                    final hospital = _filteredHospitals[index];
                    final selected = _selected?.id == hospital.id;
                    return ListTile(
                      dense: true,
                      contentPadding: const EdgeInsets.symmetric(horizontal: 4),
                      leading: Icon(
                        selected ? Icons.radio_button_checked : Icons.radio_button_off,
                        color: selected ? theme.colorScheme.primary : Colors.black45,
                      ),
                      title: Text(
                        hospital.name,
                        style: const TextStyle(fontWeight: FontWeight.w700),
                      ),
                      subtitle: Text(hospital.addressShort),
                      onTap: () => setState(() => _selected = hospital),
                    );
                  },
                ),
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: _selected == null ? null : () => Navigator.of(context).pop(_selected),
          child: const Text('Select'),
        ),
      ],
    );
  }
}

import 'package:flutter/material.dart';

import '../features/home/demo_hospitals.dart';
import '../features/home/empty_rooms.dart';
import '../features/home/patient_search_bar.dart';
import '../features/home/plus_sign.dart';
import '../features/home/select_hospital_button.dart';
import '../features/patient_detail/alert_system.dart';
import '../services/patient_repository.dart';
import '../theme/app_colors.dart';
import 'role_selection_screen.dart';

class HomePage extends StatefulWidget {
  const HomePage({
    required this.role,
    this.sessionPatientId,
    super.key,
  });

  final AppRole role;

  /// When [role] is patient, only this patient ID may be viewed.
  final String? sessionPatientId;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late DemoHospital _selectedHospital = demoHospitals.first;
  late List<String> _emptyRooms =
      List<String>.from(_selectedHospital.emptyIcuRooms);
  List<PatientRecord> _patients = [];
  bool _loading = true;
  String? _loadError;
  bool _openedPatientProfile = false;

  final TextEditingController _searchController = TextEditingController();
  String? _selectedCondition;
  String? _selectedDoctor;
  String? _selectedUnit;

  @override
  void initState() {
    super.initState();
    _loadPatients();
  }

  Future<void> _loadPatients({bool forceRefresh = false}) async {
    setState(() {
      _loading = true;
      _loadError = null;
    });
    try {
      final loaded = await PatientRepository.instance.loadPatients(
        forceRefresh: forceRefresh,
      );
      if (!mounted) return;
      var patients = loaded;
      if (!_isClinician && widget.sessionPatientId != null) {
        patients = patients
            .where((p) => p.id == widget.sessionPatientId)
            .toList();
      }
      if (!mounted) return;
      setState(() {
        _patients = patients;
        _loading = false;
      });
      final apiWarning = PatientRepository.instance.lastShockApiWarning;
      if (apiWarning != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(apiWarning),
            duration: const Duration(seconds: 6),
          ),
        );
      }
      if (!_isClinician && patients.length == 1) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted || _openedPatientProfile) return;
          _openedPatientProfile = true;
          showPatientDetails(
            context,
            patients.first,
            role: widget.role,
            sessionPatientId: widget.sessionPatientId,
          );
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loadError = e.toString();
        _loading = false;
      });
    }
  }

  void _setHospital(DemoHospital hospital) {
    setState(() {
      _selectedHospital = hospital;
      _emptyRooms = List<String>.from(hospital.emptyIcuRooms);
      _searchController.clear();
      _selectedCondition = null;
      _selectedDoctor = null;
      _selectedUnit = null;
    });
  }

  List<PatientRecord> get _filteredPatients {
    final query = _searchController.text.trim().toLowerCase();
    return _patients.where((p) {
      final parts = p.name.toLowerCase().split(RegExp(r'\s+'));
      final matchesName =
          query.isEmpty || parts.any((part) => part.contains(query));
      final matchesCondition =
          _selectedCondition == null || p.condition == _selectedCondition;
      final matchesDoctor =
          _selectedDoctor == null || p.primaryDoctor == _selectedDoctor;
      final matchesUnit =
          _selectedUnit == null || p.roomNumber.contains(_selectedUnit!);
      return matchesName && matchesCondition && matchesDoctor && matchesUnit;
    }).toList();
  }

  bool get _hasActiveFilters =>
      _selectedCondition != null ||
      _selectedDoctor != null ||
      _selectedUnit != null;

  bool get _isClinician => widget.role == AppRole.clinician;

  PatientRecord? get _sessionPatient {
    if (widget.sessionPatientId == null || _patients.isEmpty) return null;
    try {
      return _patients.firstWhere((p) => p.id == widget.sessionPatientId);
    } catch (_) {
      return null;
    }
  }

  Future<void> _openFilterSheet() async {
    final conditions =
        _patients.map((p) => p.condition).toSet().toList()..sort();
    final doctors = _patients
        .map((p) => p.primaryDoctor)
        .whereType<String>()
        .toSet()
        .toList()
      ..sort();
    final units = _patients
        .map((p) {
          final parts = p.roomNumber.split(' ');
          return parts.isNotEmpty ? parts.last : p.roomNumber;
        })
        .toSet()
        .toList()
      ..sort();

    final result = await showModalBottomSheet<_FilterState>(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => _PatientFilterSheet(
        conditions: conditions,
        doctors: doctors,
        units: units,
        initialCondition: _selectedCondition,
        initialDoctor: _selectedDoctor,
        initialUnit: _selectedUnit,
      ),
    );

    if (result != null) {
      setState(() {
        _selectedCondition = result.condition;
        _selectedDoctor = result.doctor;
        _selectedUnit = result.unit;
      });
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  void _addPatient(PatientRecord patient) {
    setState(() {
      _patients.insert(0, patient);
      for (var i = 0; i < _patients.length; i++) {
        _patients[i] = _patients[i].copyWithRank(i + 1);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final critical =
        _patients.where((p) => p.condition == 'Critical').length;
    final watch = _patients.where((p) => p.condition == 'Moderate').length;
    final stable = _patients.where((p) => p.condition == 'Stable').length;

    return Scaffold(
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _PineAppHeader(
            roleLabel: _isClinician
                ? 'Clinician view'
                : _sessionPatient?.name ?? 'Patient view',
            onSwitchRole: () {
              Navigator.of(context).pushReplacement(
                MaterialPageRoute<void>(
                  builder: (_) => const RoleSelectionScreen(),
                ),
              );
            },
          ),
          Container(
            color: const Color(0xFFF2F2F2),
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _isClinician
                      ? 'Patient priority list'
                      : 'My care dashboard',
                  style: const TextStyle(
                    color: Colors.black,
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                if (!_isClinician && _sessionPatient != null) ...[
                  const SizedBox(height: 6),
                  Text(
                    'Signed in as ${_sessionPatient!.name} · Room ${_sessionPatient!.roomNumber}',
                    style: const TextStyle(
                      fontSize: 13,
                      color: Colors.black54,
                    ),
                  ),
                ],
                const SizedBox(height: 10),
                if (_isClinician)
                  Row(
                    children: [
                      _SummaryChip(
                        label: 'Critical',
                        count: critical,
                        color: const Color(0xFFE05A5A),
                      ),
                      const SizedBox(width: 8),
                      _SummaryChip(
                        label: 'Watch',
                        count: watch,
                        color: const Color(0xFFD4A030),
                      ),
                      const SizedBox(width: 8),
                      _SummaryChip(
                        label: 'Stable',
                        count: stable,
                        color: const Color(0xFF4A9E6A),
                      ),
                    ],
                  ),
                if (_isClinician) ...[
                  const SizedBox(height: 12),
                  PatientSearchBar(
                    controller: _searchController,
                    onChanged: (_) => setState(() {}),
                    onFilterTap: _openFilterSheet,
                    isFilterActive: _hasActiveFilters,
                  ),
                ],
                const SizedBox(height: 16),
              ],
            ),
          ),
          Expanded(
            child: Container(
              color: const Color(0xFFF2F2F2),
              child: _loading
                  ? const Center(child: CircularProgressIndicator())
                  : _loadError != null
                      ? Center(
                          child: Padding(
                            padding: const EdgeInsets.all(24),
                            child: Text(
                              'Could not load patient data.\n$_loadError',
                              textAlign: TextAlign.center,
                            ),
                          ),
                        )
                      : _filteredPatients.isEmpty
                          ? Center(
                              child: Padding(
                                padding: const EdgeInsets.all(24),
                                child: Text(
                                  _isClinician
                                      ? 'No patients match your filters.'
                                      : 'Your profile could not be loaded. '
                                          'Go back and select your name again.',
                                  textAlign: TextAlign.center,
                                ),
                              ),
                            )
                          : ListView.separated(
                              padding:
                                  const EdgeInsets.fromLTRB(16, 0, 16, 16),
                              itemCount: _filteredPatients.length,
                              separatorBuilder: (_, __) =>
                                  const SizedBox(height: 12),
                              itemBuilder: (context, index) {
                                final patient = _filteredPatients[index];
                                if (!_isClinician &&
                                    widget.sessionPatientId != null &&
                                    patient.id != widget.sessionPatientId) {
                                  return const SizedBox.shrink();
                                }
                                return PatientListCard(
                                  patient: patient,
                                  isClinician: _isClinician,
                                  onTap: () => showPatientDetails(
                                    context,
                                    patient,
                                    role: widget.role,
                                    sessionPatientId:
                                        widget.sessionPatientId,
                                  ),
                                );
                              },
                            ),
            ),
          ),
        ],
      ),
      bottomNavigationBar: SafeArea(
        top: false,
        child: Container(
          decoration: const BoxDecoration(
            color: Colors.white,
            border: Border(top: BorderSide(color: Colors.black12)),
          ),
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
          child: Row(
            children: [
              if (_isClinician)
                SelectHospitalButton(
                  selectedHospital: _selectedHospital,
                  onHospitalSelected: _setHospital,
                )
              else
                Expanded(
                  child: Text(
                    'Viewing your record only',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.black.withValues(alpha: 0.5),
                    ),
                  ),
                ),
              if (_isClinician) ...[
                const Spacer(),
                PlusSignButton(onPatientCreated: _addPatient),
                const Spacer(),
                EmptyRoomsButton(emptyRooms: _emptyRooms),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _SummaryChip extends StatelessWidget {
  const _SummaryChip({
    required this.label,
    required this.count,
    required this.color,
  });
  final String label;
  final int count;
  final Color color;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(shape: BoxShape.circle, color: color),
            ),
            const SizedBox(width: 6),
            Text(
              '$count $label',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: color,
              ),
            ),
          ],
        ),
      );
}

class _PineAppHeader extends StatelessWidget {
  const _PineAppHeader({
    required this.roleLabel,
    required this.onSwitchRole,
  });

  final String roleLabel;
  final VoidCallback onSwitchRole;

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.paddingOf(context).top;
    return Container(
      width: double.infinity,
      padding: EdgeInsets.fromLTRB(16, top + 12, 16, 16),
      decoration: const BoxDecoration(
        color: BhColors.ink,
        boxShadow: [
          BoxShadow(
            color: Color(0x33000000),
            blurRadius: 12,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Image.asset(
            'assets/baptist_logo.png',
            height: 44,
            fit: BoxFit.contain,
            filterQuality: FilterQuality.high,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Baptist Health Cardiogenic Shock Tracker',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                        fontFamily: 'Georgia',
                        letterSpacing: 0.2,
                        fontSize: 16,
                      ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Cardiogenic shock · $roleLabel',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.white.withValues(alpha: 0.78),
                        fontWeight: FontWeight.w500,
                      ),
                ),
              ],
            ),
          ),
          TextButton(
            onPressed: onSwitchRole,
            style: TextButton.styleFrom(foregroundColor: Colors.white70),
            child: const Text('Switch'),
          ),
        ],
      ),
    );
  }
}

class _FilterState {
  const _FilterState({this.condition, this.doctor, this.unit});
  final String? condition;
  final String? doctor;
  final String? unit;
}

class _PatientFilterSheet extends StatefulWidget {
  const _PatientFilterSheet({
    required this.conditions,
    required this.doctors,
    required this.units,
    required this.initialCondition,
    required this.initialDoctor,
    required this.initialUnit,
  });
  final List<String> conditions;
  final List<String> doctors;
  final List<String> units;
  final String? initialCondition;
  final String? initialDoctor;
  final String? initialUnit;

  @override
  State<_PatientFilterSheet> createState() => _PatientFilterSheetState();
}

class _PatientFilterSheetState extends State<_PatientFilterSheet> {
  late String? _condition = widget.initialCondition;
  late String? _doctor = widget.initialDoctor;
  late String? _unit = widget.initialUnit;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.fromLTRB(
        16,
        16,
        16,
        MediaQuery.of(context).viewInsets.bottom + 20,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Filter patients',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
          const SizedBox(height: 16),
          DropdownButtonFormField<String?>(
            initialValue: _condition,
            decoration: const InputDecoration(
                labelText: 'Risk tier', border: OutlineInputBorder()),
            items: [
              const DropdownMenuItem<String?>(
                  value: null, child: Text('All tiers')),
              ...widget.conditions.map((c) =>
                  DropdownMenuItem<String?>(value: c, child: Text(c))),
            ],
            onChanged: (v) => setState(() => _condition = v),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String?>(
            initialValue: _doctor,
            decoration: const InputDecoration(
                labelText: 'Doctor', border: OutlineInputBorder()),
            items: [
              const DropdownMenuItem<String?>(
                  value: null, child: Text('All doctors')),
              ...widget.doctors.map((d) =>
                  DropdownMenuItem<String?>(value: d, child: Text(d))),
            ],
            onChanged: (v) => setState(() => _doctor = v),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String?>(
            initialValue: _unit,
            decoration: const InputDecoration(
                labelText: 'ICU unit', border: OutlineInputBorder()),
            items: [
              const DropdownMenuItem<String?>(
                  value: null, child: Text('All ICU units')),
              ...widget.units.map((u) =>
                  DropdownMenuItem<String?>(value: u, child: Text(u))),
            ],
            onChanged: (v) => setState(() => _unit = v),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              TextButton(
                onPressed: () => setState(() {
                  _condition = null;
                  _doctor = null;
                  _unit = null;
                }),
                child: const Text('Clear'),
              ),
              const Spacer(),
              FilledButton(
                onPressed: () => Navigator.pop(
                  context,
                  _FilterState(
                    condition: _condition,
                    doctor: _doctor,
                    unit: _unit,
                  ),
                ),
                child: const Text('Apply filters'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';

import '../features/home/demo_hospitals.dart';
import '../features/home/empty_rooms.dart';
import '../features/home/patient_search_bar.dart';
import '../features/home/select_hospital_button.dart';
import '../features/patient_detail/alert_system.dart';
import '../services/patient_repository.dart';
import '../theme/app_colors.dart';
import '../widgets/bh_branded_header.dart';
import 'role_selection_screen.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

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
  final TextEditingController _searchController = TextEditingController();
  String? _selectedCondition;
  String? _selectedDoctor;
  String? _selectedUnit;
  String? _selectedScaiStage;
  _EscalationFilter _selectedEscalation = _EscalationFilter.all;

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
      _selectedScaiStage = null;
      _selectedEscalation = _EscalationFilter.all;
    });
  }

  List<PatientRecord> get _hospitalPatients => _patients
      .where((p) => p.demoHospitalId == _selectedHospital.id)
      .toList();

  List<PatientRecord> get _filteredPatients {
    final query = _searchController.text.trim().toLowerCase();
    return _patients.where((p) {
      final matchesHospital = p.demoHospitalId == _selectedHospital.id;
      final parts = p.name.toLowerCase().split(RegExp(r'\s+'));
      final matchesName =
          query.isEmpty || parts.any((part) => part.contains(query));
      final matchesCondition =
          _selectedCondition == null || p.condition == _selectedCondition;
      final matchesDoctor =
          _selectedDoctor == null || p.primaryDoctor == _selectedDoctor;
      final matchesUnit =
          _selectedUnit == null || p.roomNumber.contains(_selectedUnit!);
      final matchesScai = _selectedScaiStage == null ||
          p.scaiStageCurrent == _selectedScaiStage;
      final matchesEscalation = _matchesEscalationFilter(p);
      return matchesHospital &&
          matchesName &&
          matchesCondition &&
          matchesDoctor &&
          matchesUnit &&
          matchesScai &&
          matchesEscalation;
    }).toList();
  }

  bool _matchesEscalationFilter(PatientRecord p) {
    final mcsLikely =
        p.predictions.mcs12hNeeded && !p.mechanicalSupport.onMcs;
    final ecmoLikely =
        p.predictions.vaEcmo12hNeeded && !p.mechanicalSupport.onVaEcmo;
    switch (_selectedEscalation) {
      case _EscalationFilter.all:
        return true;
      case _EscalationFilter.mcsLikely:
        return mcsLikely;
      case _EscalationFilter.ecmoLikely:
        return ecmoLikely;
      case _EscalationFilter.mcsOrEcmoLikely:
        return mcsLikely || ecmoLikely;
    }
  }

  bool get _hasActiveFilters =>
      _selectedCondition != null ||
      _selectedDoctor != null ||
      _selectedUnit != null ||
      _selectedScaiStage != null ||
      _selectedEscalation != _EscalationFilter.all;

  Future<void> _openFilterSheet() async {
    final pool = _hospitalPatients;
    final conditions = pool.map((p) => p.condition).toSet().toList()..sort();
    final doctors = pool
        .map((p) => p.primaryDoctor)
        .whereType<String>()
        .toSet()
        .toList()
      ..sort();
    final units = pool
        .map((p) {
          final parts = p.roomNumber.split(' ');
          return parts.isNotEmpty ? parts.last : p.roomNumber;
        })
        .toSet()
        .toList()
      ..sort();
    const scaiOrder = ['A', 'B', 'C', 'D', 'E'];
    final scaiStages = pool
        .map((p) => p.scaiStageCurrent)
        .whereType<String>()
        .toSet()
        .toList()
      ..sort((a, b) {
        final ia = scaiOrder.indexOf(a);
        final ib = scaiOrder.indexOf(b);
        return (ia < 0 ? 99 : ia).compareTo(ib < 0 ? 99 : ib);
      });

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
        scaiStages: scaiStages,
        initialCondition: _selectedCondition,
        initialDoctor: _selectedDoctor,
        initialUnit: _selectedUnit,
        initialScaiStage: _selectedScaiStage,
        initialEscalation: _selectedEscalation,
      ),
    );

    if (result != null) {
      setState(() {
        _selectedCondition = result.condition;
        _selectedDoctor = result.doctor;
        _selectedUnit = result.unit;
        _selectedScaiStage = result.scaiStage;
        _selectedEscalation = result.escalation;
      });
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
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
          BhBrandedHeader(
            subtitle: 'Cardiogenic shock · Clinician view',
            trailing: TextButton(
              onPressed: () {
                Navigator.of(context).pushReplacement(
                  MaterialPageRoute<void>(
                    builder: (_) => const RoleSelectionScreen(),
                  ),
                );
              },
              style: TextButton.styleFrom(foregroundColor: Colors.white70),
              child: const Text('Switch View'),
            ),
          ),
          Container(
            color: const Color(0xFFF2F2F2),
            padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Patient priority list',
                  style: TextStyle(
                    color: Colors.black,
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 10),
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
                const SizedBox(height: 12),
                PatientSearchBar(
                  controller: _searchController,
                  onChanged: (_) => setState(() {}),
                  onFilterTap: _openFilterSheet,
                  isFilterActive: _hasActiveFilters,
                ),
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
                          ? const Center(
                              child: Padding(
                                padding: EdgeInsets.all(24),
                                child: Text(
                                  'No patients match your filters.',
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
                                return PatientListCard(
                                  patient: patient,
                                  isClinician: true,
                                  onTap: () => showPatientDetails(
                                    context,
                                    patient,
                                    role: AppRole.clinician,
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
              SelectHospitalButton(
                selectedHospital: _selectedHospital,
                onHospitalSelected: _setHospital,
              ),
              const Spacer(),
              EmptyRoomsButton(emptyRooms: _emptyRooms),
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

enum _EscalationFilter {
  all,
  mcsLikely,
  ecmoLikely,
  mcsOrEcmoLikely,
}

class _FilterState {
  const _FilterState({
    this.condition,
    this.doctor,
    this.unit,
    this.scaiStage,
    this.escalation = _EscalationFilter.all,
  });
  final String? condition;
  final String? doctor;
  final String? unit;
  final String? scaiStage;
  final _EscalationFilter escalation;
}

class _PatientFilterSheet extends StatefulWidget {
  const _PatientFilterSheet({
    required this.conditions,
    required this.doctors,
    required this.units,
    required this.scaiStages,
    required this.initialCondition,
    required this.initialDoctor,
    required this.initialUnit,
    required this.initialScaiStage,
    required this.initialEscalation,
  });
  final List<String> conditions;
  final List<String> doctors;
  final List<String> units;
  final List<String> scaiStages;
  final String? initialCondition;
  final String? initialDoctor;
  final String? initialUnit;
  final String? initialScaiStage;
  final _EscalationFilter initialEscalation;

  @override
  State<_PatientFilterSheet> createState() => _PatientFilterSheetState();
}

class _PatientFilterSheetState extends State<_PatientFilterSheet> {
  late String? _condition = widget.initialCondition;
  late String? _doctor = widget.initialDoctor;
  late String? _unit = widget.initialUnit;
  late String? _scaiStage = widget.initialScaiStage;
  late _EscalationFilter _escalation = widget.initialEscalation;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.fromLTRB(
        16,
        16,
        16,
        MediaQuery.of(context).viewInsets.bottom + 20,
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Filter patients',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String?>(
              initialValue: _scaiStage,
              decoration: const InputDecoration(
                labelText: 'SCAI stage',
                border: OutlineInputBorder(),
              ),
              items: [
                const DropdownMenuItem<String?>(
                  value: null,
                  child: Text('All SCAI stages'),
                ),
                ...widget.scaiStages.map(
                  (s) => DropdownMenuItem<String?>(
                    value: s,
                    child: Text('SCAI $s'),
                  ),
                ),
              ],
              onChanged: (v) => setState(() => _scaiStage = v),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<_EscalationFilter>(
              initialValue: _escalation,
              decoration: const InputDecoration(
                labelText: 'MCS / VA-ECMO (12h)',
                border: OutlineInputBorder(),
              ),
              items: const [
                DropdownMenuItem(
                  value: _EscalationFilter.all,
                  child: Text('All — any escalation risk'),
                ),
                DropdownMenuItem(
                  value: _EscalationFilter.mcsLikely,
                  child: Text('MCS likely within 12 hours'),
                ),
                DropdownMenuItem(
                  value: _EscalationFilter.ecmoLikely,
                  child: Text('VA-ECMO likely within 12 hours'),
                ),
                DropdownMenuItem(
                  value: _EscalationFilter.mcsOrEcmoLikely,
                  child: Text('MCS or VA-ECMO likely within 12 hours'),
                ),
              ],
              onChanged: (v) => setState(
                () => _escalation = v ?? _EscalationFilter.all,
              ),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String?>(
              initialValue: _condition,
              decoration: const InputDecoration(
                labelText: 'Risk tier',
                border: OutlineInputBorder(),
              ),
              items: [
                const DropdownMenuItem<String?>(
                  value: null,
                  child: Text('All tiers'),
                ),
                ...widget.conditions.map(
                  (c) => DropdownMenuItem<String?>(value: c, child: Text(c)),
                ),
              ],
              onChanged: (v) => setState(() => _condition = v),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String?>(
              initialValue: _doctor,
              decoration: const InputDecoration(
                labelText: 'Doctor',
                border: OutlineInputBorder(),
              ),
              items: [
                const DropdownMenuItem<String?>(
                  value: null,
                  child: Text('All doctors'),
                ),
                ...widget.doctors.map(
                  (d) => DropdownMenuItem<String?>(value: d, child: Text(d)),
                ),
              ],
              onChanged: (v) => setState(() => _doctor = v),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String?>(
              initialValue: _unit,
              decoration: const InputDecoration(
                labelText: 'ICU unit',
                border: OutlineInputBorder(),
              ),
              items: [
                const DropdownMenuItem<String?>(
                  value: null,
                  child: Text('All ICU units'),
                ),
                ...widget.units.map(
                  (u) => DropdownMenuItem<String?>(value: u, child: Text(u)),
                ),
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
                    _scaiStage = null;
                    _escalation = _EscalationFilter.all;
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
                      scaiStage: _scaiStage,
                      escalation: _escalation,
                    ),
                  ),
                  child: const Text('Apply filters'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

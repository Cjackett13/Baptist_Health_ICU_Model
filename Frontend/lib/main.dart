// lib/main.dart
import 'dart:math';
import 'package:flutter/material.dart';
import 'alert_system.dart';
import 'demo_hospitals.dart';
import 'empty_rooms.dart';
import 'patient_search_bar.dart';
import 'plus_sign.dart';
import 'select_hospital_button.dart';

void main() {
  runApp(const MyApp());
}

abstract final class BhColors {
  static const primary = Color(0xFF7BC74D);
  static const ink = Color(0xFF222831);
  static const slate = Color(0xFF393E46);
  static const background = Color(0xFFEEEEEE);
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Baptist Health ICU',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: BhColors.background,
        colorScheme: const ColorScheme.light(
          primary: BhColors.primary,
          onPrimary: Colors.white,
          surface: Colors.white,
          onSurface: BhColors.ink,
          secondary: BhColors.slate,
          onSecondary: Colors.white,
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  late DemoHospital _selectedHospital = demoHospitals.first;
  late List<String> _emptyRooms =
      List<String>.from(_selectedHospital.emptyIcuRooms);
  late List<PatientRecord> _patients =
      _buildSeedPatientsForHospital(_selectedHospital);

  final TextEditingController _searchController = TextEditingController();
  String? _selectedCondition;
  String? _selectedDoctor;
  String? _selectedUnit;

  void _setHospital(DemoHospital hospital) {
    setState(() {
      _selectedHospital = hospital;
      _emptyRooms = List<String>.from(hospital.emptyIcuRooms);
      _patients = _buildSeedPatientsForHospital(hospital);
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
      final matchesName = query.isEmpty ||
          parts.any((part) => part.contains(query));
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
    // Summary counts for header strip
    final critical =
        _patients.where((p) => p.condition == 'Critical').length;
    final watch =
        _patients.where((p) => p.condition == 'Moderate').length;
    final stable =
        _patients.where((p) => p.condition == 'Stable').length;

    return Scaffold(
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _PineAppHeader(),
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
                // Risk summary strip
                Row(
                  children: [
                    _SummaryChip(
                        label: 'Critical',
                        count: critical,
                        color: const Color(0xFFE05A5A)),
                    const SizedBox(width: 8),
                    _SummaryChip(
                        label: 'Watch',
                        count: watch,
                        color: const Color(0xFFD4A030)),
                    const SizedBox(width: 8),
                    _SummaryChip(
                        label: 'Stable',
                        count: stable,
                        color: const Color(0xFF4A9E6A)),
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
              child: ListView.separated(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                itemCount: _filteredPatients.length,
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  final patient = _filteredPatients[index];
                  return PatientListCard(
                    patient: patient,
                    onTap: () => showPatientDetails(context, patient),
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
              PlusSignButton(onPatientCreated: _addPatient),
              const Spacer(),
              EmptyRoomsButton(emptyRooms: _emptyRooms),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SUMMARY CHIP
// ─────────────────────────────────────────────────────────────────────────────
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
        padding:
            const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
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
              decoration:
                  BoxDecoration(shape: BoxShape.circle, color: color),
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

// ─────────────────────────────────────────────────────────────────────────────
// SEED PATIENT GENERATOR
// Sorted by icuTransferRisk descending — highest risk at top.
// Replace with real Firestore fetch when backend is ready.
// ─────────────────────────────────────────────────────────────────────────────
List<PatientRecord> _buildSeedPatientsForHospital(DemoHospital hospital) {
  const firstNames = [
    'Jordan', 'Layla', 'Rajiv', 'Amelia', 'Marcus',
    'Noah', 'Avery', 'Elijah', 'Mia', 'Liam',
    'Sofia', 'Ethan', 'Isabella', 'Lucas', 'Emma',
    'Aiden', 'Olivia', 'Jackson', 'Ava', 'Logan',
  ];
  const lastNames = [
    'Smith', 'Torres', 'Kumar', 'Johnson', 'Reed',
    'Nguyen', 'Carter', 'Patel', 'Brooks', 'Diaz',
    'Kim', 'Okafor', 'Martinez', 'Chen', 'Williams',
    'Garcia', 'Brown', 'Davis', 'Wilson', 'Moore',
  ];
  const doctors = [
    'Dr. Maya Chen', 'Dr. David Patel', 'Dr. Naomi Lee',
    'Dr. Jordan Ng', 'Dr. Emily Brooks', 'Dr. Samuel Rivera',
  ];
  const diagnoses = [
    'CHF', 'ACS', 'Afib', 'HTN', 'COPD',
    'Pneumonia', 'Sepsis', 'MI', 'PE', 'DVT',
  ];
  const genders = ['M', 'F'];
  const issues = [
    'Acute respiratory distress with elevated heart rate',
    'Sepsis and unstable blood pressure',
    'Post-operative respiratory support',
    'Recovering from pneumonia with O2 support',
    'Routine monitoring and vital stabilisation',
    'Cardiac rhythm irregularities under observation',
    'Fluid overload with reduced ejection fraction',
    'Uncontrolled hypertension post-procedure',
  ];

  final roomPool = hospital.emptyIcuRooms.isEmpty
      ? const ['ICU 1A', 'ICU 1B', 'ICU 1C']
      : hospital.emptyIcuRooms;

  final salt =
      hospital.id.codeUnits.fold<int>(0, (acc, c) => acc + c);
  final rng = Random(salt);
  final patients = <PatientRecord>[];

  for (var i = 0; i < 20; i++) {
    final firstName = firstNames[(i + salt) % firstNames.length];
    final lastName =
        lastNames[((i * 3 + salt) ~/ 2) % lastNames.length];
    final room = roomPool[(i + (salt % 7)) % roomPool.length];
    final age = 45 + rng.nextInt(45);
    final gender = genders[rng.nextInt(2)];
    final diagnosis = diagnoses[rng.nextInt(diagnoses.length)];
    final daysAdmitted = 1 + rng.nextInt(10);

    // Realistic risk distribution
    final double icuRisk;
    final tier = rng.nextDouble();
    if (tier < 0.15) {
      icuRisk = 0.65 + rng.nextDouble() * 0.34;
    } else if (tier < 0.50) {
      icuRisk = 0.40 + rng.nextDouble() * 0.24;
    } else {
      icuRisk = 0.05 + rng.nextDouble() * 0.34;
    }

    final features = PatientFeatures(
      numMedications: icuRisk >= 0.65
          ? 12 + rng.nextInt(20)
          : icuRisk >= 0.40
              ? 7 + rng.nextInt(12)
              : 2 + rng.nextInt(8),
      numberInpatient:
          icuRisk >= 0.65 ? 2 + rng.nextInt(6) : rng.nextInt(3),
      numLabProcedures: 20 + rng.nextInt(80),
      timeInHospital: daysAdmitted.clamp(1, 14),
      numberDiagnoses: 2 + rng.nextInt(8),
      numberEmergency:
          icuRisk >= 0.65 ? rng.nextInt(4) : rng.nextInt(2),
      numberOutpatient: rng.nextInt(5),
      ageMid: age.toDouble(),
    );

    patients.add(
      PatientRecord(
        rank: i + 1,
        name: '$firstName $lastName',
        id: 'P-${30000 + i + (salt % 900)}',
        roomNumber: room,
        predictions: generateMockPredictions(icuRisk, features, rng),
        features: features,
        primaryDoctor: doctors[i % doctors.length],
        issue: issues[i % issues.length],
        age: age,
        gender: gender,
        diagnosis: diagnosis,
        daysAdmitted: daysAdmitted,
      ),
    );
  }

  // Sort highest ICU risk first
  patients.sort(
      (a, b) => b.predictions.icuTransferRisk
          .compareTo(a.predictions.icuTransferRisk));

  // Re-rank after sort
  return List.generate(
    patients.length,
    (i) => patients[i].copyWithRank(i + 1),
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// HEADER
// ─────────────────────────────────────────────────────────────────────────────
class _PineAppHeader extends StatelessWidget {
  const _PineAppHeader();

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
                  'Baptist Health',
                  style:
                      Theme.of(context).textTheme.titleLarge?.copyWith(
                            color: Colors.white,
                            fontWeight: FontWeight.w700,
                            fontFamily: 'Georgia',
                            letterSpacing: 0.2,
                          ),
                ),
                const SizedBox(height: 2),
                Text(
                  'ICU planning workspace',
                  style:
                      Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.white.withValues(alpha: 0.78),
                            fontWeight: FontWeight.w500,
                          ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// FILTER SHEET
// ─────────────────────────────────────────────────────────────────────────────
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
        16, 16, 16,
        MediaQuery.of(context).viewInsets.bottom + 20,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Filter patients',
              style:
                  TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
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
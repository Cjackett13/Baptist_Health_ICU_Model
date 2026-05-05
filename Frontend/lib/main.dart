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

/// Baptist Health PineApp–inspired palette (provided hex).
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
  late List<String> _emptyRooms = List<String>.from(_selectedHospital.emptyIcuRooms);
  late List<PatientRecord> _patients = _buildSeedPatientsForHospital(_selectedHospital);

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
    return _patients.where((patient) {
      final parts = patient.name.toLowerCase().split(RegExp(r'\s+'));
      final firstName = parts.isNotEmpty ? parts.first : '';
      final lastName = parts.length > 1 ? parts.last : '';
      final matchesName =
          query.isEmpty || firstName.contains(query) || lastName.contains(query);

      final matchesCondition =
          _selectedCondition == null || patient.condition == _selectedCondition;

      final matchesDoctor =
          _selectedDoctor == null || patient.primaryDoctor == _selectedDoctor;

      final matchesUnit =
          _selectedUnit == null || patient.roomNumber.contains(_selectedUnit!);

      return matchesName && matchesCondition && matchesDoctor && matchesUnit;
    }).toList();
  }

  bool get _hasActiveFilters =>
      _selectedCondition != null || _selectedDoctor != null || _selectedUnit != null;

  Future<void> _openFilterSheet() async {
    final conditions = _patients.map((p) => p.condition).toSet().toList()..sort();
    final doctors = _patients
        .map((p) => p.primaryDoctor)
        .whereType<String>()
        .toSet()
        .toList()
      ..sort();
    final units = _patients
        .map((p) {
          final roomParts = p.roomNumber.split(' ');
          return roomParts.isNotEmpty ? roomParts.last : p.roomNumber;
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
      _patients.insert(
        0,
        PatientRecord(
          rank: 1,
          name: patient.name,
          id: patient.id,
          condition: patient.condition,
          roomNumber: patient.roomNumber,
          primaryDoctor: patient.primaryDoctor,
          issue: patient.issue,
        ),
      );

      for (var i = 0; i < _patients.length; i++) {
        final current = _patients[i];
        _patients[i] = PatientRecord(
          rank: i + 1,
          name: current.name,
          id: current.id,
          condition: current.condition,
          roomNumber: current.roomNumber,
          primaryDoctor: current.primaryDoctor,
          issue: current.issue,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _PineAppHeader(),
          Container(
            color: const Color(0xFFF2F2F2),
            padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
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
                separatorBuilder: (context, index) =>
                    const SizedBox(height: 12),
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
          decoration: BoxDecoration(
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
              SizedBox(
                width: MediaQuery.of(context).size.width * 0.4,
                child: EmptyRoomsButton(emptyRooms: _emptyRooms),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

List<PatientRecord> _buildSeedPatientsForHospital(DemoHospital hospital) {
  const firstNames = <String>[
    'Jordan',
    'Layla',
    'Rajiv',
    'Amelia',
    'Marcus',
    'Noah',
    'Avery',
    'Elijah',
    'Mia',
    'Liam',
  ];
  const lastNames = <String>[
    'Smith',
    'Torres',
    'Kumar',
    'Johnson',
    'Reed',
    'Nguyen',
    'Carter',
    'Patel',
    'Brooks',
    'Diaz',
  ];
  const doctors = <String>[
    'Dr. Maya Chen',
    'Dr. David Patel',
    'Dr. Naomi Lee',
    'Dr. Jordan Ng',
    'Dr. Emily Brooks',
    'Dr. Samuel Rivera',
  ];
  const issues = <String>[
    'Acute respiratory distress with elevated heart rate',
    'Sepsis and unstable blood pressure',
    'Post-operative respiratory support',
    'Recovering from pneumonia',
    'Routine monitoring and support',
    'Cardiac rhythm irregularities under observation',
  ];
  const conditions = <String>['Critical', 'Moderate', 'Stable'];

  final roomPool = hospital.emptyIcuRooms.isEmpty
      ? const <String>['ICU 1A', 'ICU 1B', 'ICU 1C']
      : hospital.emptyIcuRooms;

  final salt = hospital.id.codeUnits.fold<int>(0, (acc, c) => acc + c);
  final patients = <PatientRecord>[];
  for (var i = 0; i < 100; i++) {
    final firstName = firstNames[(i + salt) % firstNames.length];
    final lastName = lastNames[((i + (salt ~/ 3)) ~/ firstNames.length) % lastNames.length];
    final room = roomPool[(i + (salt % 7)) % roomPool.length];

    patients.add(
      PatientRecord(
        rank: i + 1,
        name: '$firstName $lastName',
        id: 'P-${30000 + i + (salt % 900)}',
        condition: conditions[i % conditions.length],
        roomNumber: room,
        primaryDoctor: doctors[i % doctors.length],
        issue: issues[i % issues.length],
      ),
    );
  }

  return patients;
}

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
            alignment: Alignment.centerLeft,
            filterQuality: FilterQuality.high,
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Baptist Health',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: Colors.white,
                    fontWeight: FontWeight.w700,
                    fontFamily: 'Georgia',
                    letterSpacing: 0.2,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'ICU planning workspace',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
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
          const Text(
            'Filter patients',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String?>(
            initialValue: _condition,
            decoration: const InputDecoration(
              labelText: 'Condition',
              border: OutlineInputBorder(),
            ),
            items: [
              const DropdownMenuItem<String?>(
                value: null,
                child: Text('All conditions'),
              ),
              ...widget.conditions.map(
                (condition) =>
                    DropdownMenuItem<String?>(value: condition, child: Text(condition)),
              ),
            ],
            onChanged: (value) => setState(() => _condition = value),
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
                (doctor) => DropdownMenuItem<String?>(
                  value: doctor,
                  child: Text(doctor),
                ),
              ),
            ],
            onChanged: (value) => setState(() => _doctor = value),
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
                (unit) => DropdownMenuItem<String?>(value: unit, child: Text(unit)),
              ),
            ],
            onChanged: (value) => setState(() => _unit = value),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              TextButton(
                onPressed: () {
                  setState(() {
                    _condition = null;
                    _doctor = null;
                    _unit = null;
                  });
                },
                child: const Text('Clear'),
              ),
              const Spacer(),
              FilledButton(
                onPressed: () {
                  Navigator.pop(
                    context,
                    _FilterState(
                      condition: _condition,
                      doctor: _doctor,
                      unit: _unit,
                    ),
                  );
                },
                child: const Text('Apply filters'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'alert_system.dart';
import 'plus_sign.dart';
import 'package:flutter_application_1/search_input.dart';

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
  static final _seedPatients = <PatientRecord>[
    const PatientRecord(
      rank: 1,
      name: 'Jordan Smith',
      id: 'P-32491',
      condition: 'Critical',
      roomNumber: 'ICU 4A',
      primaryDoctor: 'Dr. Maya Chen',
      issue: 'Acute respiratory distress with elevated heart rate',
    ),
    const PatientRecord(
      rank: 2,
      name: 'Layla Torres',
      id: 'P-45102',
      condition: 'Moderate',
      roomNumber: 'ICU 2C',
      primaryDoctor: 'Dr. David Patel',
      issue: 'Sepsis and unstable blood pressure',
    ),
    const PatientRecord(
      rank: 3,
      name: 'Rajiv Kumar',
      id: 'P-88014',
      condition: 'Moderate',
      roomNumber: 'Step-down 1B',
      primaryDoctor: 'Dr. Naomi Lee',
      issue: 'Post-operative respiratory support',
    ),
    const PatientRecord(
      rank: 4,
      name: 'Amelia Johnson',
      id: 'P-66520',
      condition: 'Moderate',
      roomNumber: 'Step-down 3D',
      primaryDoctor: 'Dr. Jordan Ng',
      issue: 'Recovering from pneumonia',
    ),
    const PatientRecord(
      rank: 5,
      name: 'Marcus Reed',
      id: 'P-99133',
      condition: 'Stable',
      roomNumber: 'Ward 5A',
      primaryDoctor: 'Dr. Emily Brooks',
      issue: 'Routine monitoring and support',
    ),
  ];
  late final TextEditingController _searchController;

  @override
  void initState() {
    super.initState();
    _searchController = TextEditingController();
    _searchController.addListener(_onSearchChanged);
  }

  void _onSearchChanged() {
    setState(() {});
  }

  @override
  void dispose() {
    _searchController.removeListener(_onSearchChanged);
    _searchController.dispose();
    super.dispose();
  }

  late final List<PatientRecord> _patients = List<PatientRecord>.from(
    _seedPatients,
  );

  List<PatientRecord> get _filteredPatients {
    final query = _searchController.text.trim().toLowerCase();
    if (query.isEmpty) return _patients;
    return _patients.where((patient) {
      return patient.name.toLowerCase().contains(query) ||
          patient.id.toLowerCase().contains(query) ||
          patient.roomNumber.toLowerCase().contains(query) ||
          patient.condition.toLowerCase().contains(query);
    }).toList();
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
          vitals: patient.vitals,
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
          vitals: current.vitals,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final filteredPatients = _filteredPatients;
    return Scaffold(
      body: Stack(
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const _PineAppHeader(),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
                child: SearchInput(
                  textController: _searchController,
                  hintText: 'Search patients…',
                ),
              ),
              Expanded(
                child: ListView.builder(
                  padding: const EdgeInsets.fromLTRB(16, 4, 16, 100),
                  itemCount: filteredPatients.length,
                  itemBuilder: (context, index) {
                    final patient = filteredPatients[index];
                    return Container(
                      margin: const EdgeInsets.only(bottom: 10),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: BhColors.slate.withValues(alpha: 0.15),
                        ),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          CircleAvatar(
                            radius: 14,
                            backgroundColor: BhColors.primary.withValues(alpha: 0.2),
                            child: Text(
                              '${patient.rank}',
                              style: const TextStyle(
                                color: BhColors.ink,
                                fontWeight: FontWeight.w700,
                                fontSize: 12,
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  patient.name,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w700,
                                    fontSize: 15,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  '${patient.id}  •  ${patient.roomNumber}',
                                  style: TextStyle(
                                    color: BhColors.ink.withValues(alpha: 0.7),
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  patient.condition,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w600,
                                    fontSize: 12,
                                  ),
                                ),
                                if (patient.issue != null) ...[
                                  const SizedBox(height: 4),
                                  Text(
                                    patient.issue!,
                                    style: TextStyle(
                                      color: BhColors.ink.withValues(alpha: 0.75),
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
          Align(
            alignment: Alignment.bottomRight,
            child: SafeArea(
              top: false,
              child: Padding(
                padding: const EdgeInsets.only(right: 14, bottom: 14),
                child: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(12),
                    boxShadow: [
                      BoxShadow(
                        color: BhColors.ink.withValues(alpha: 0.08),
                        blurRadius: 16,
                        offset: const Offset(0, 6),
                      ),
                    ],
                    border: Border.all(
                      color: BhColors.slate.withValues(alpha: 0.12),
                    ),
                  ),
                  child: PlusSignButton(
                    onPatientCreated: _addPatient,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
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

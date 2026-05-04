import 'package:flutter/material.dart';
import 'alert_system.dart';

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

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  static final _patients = <PatientRecord>[
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
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: Colors.black12),
                  ),
                  child: const TextField(
                    decoration: InputDecoration(
                      prefixIcon: Icon(Icons.search),
                      hintText: 'Search patients',
                      border: InputBorder.none,
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 16,
                      ),
                    ),
                  ),
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
                itemCount: _patients.length,
                separatorBuilder: (context, index) =>
                    const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  final patient = _patients[index];
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
              SizedBox(
                width: MediaQuery.of(context).size.width * 0.4,
                height: 52,
                child: FilledButton(
                  onPressed: () {},
                  style: FilledButton.styleFrom(
                    backgroundColor: BhColors.slate,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text('Select hospital'),
                ),
              ),
              const Spacer(),
            ],
          ),
        ),
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

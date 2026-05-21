/// Demo hospital sites. Seed export assigns [patientsPerHospital] real
/// parquet encounters per [id] (see `scripts/export_patient_seed.py`).
const int patientsPerHospital = 10;

class DemoHospital {
  const DemoHospital({
    required this.id,
    required this.name,
    required this.addressLine1,
    required this.city,
    required this.state,
    required this.zip,
    required this.emptyIcuRooms,
  });

  final String id;
  final String name;
  final String addressLine1;
  final String city;
  final String state;
  final String zip;
  final List<String> emptyIcuRooms;

  String get addressShort => '$addressLine1, $city, $state $zip';
}

const demoHospitals = <DemoHospital>[
  DemoHospital(
    id: 'bh-jax',
    name: 'Baptist Health Jacksonville (Demo)',
    addressLine1: '800 Prudential Dr',
    city: 'Jacksonville',
    state: 'FL',
    zip: '32207',
    emptyIcuRooms: <String>[
      'ICU 1A',
      'ICU 1C',
      'ICU 2B',
      'ICU 3A',
      'ICU 4D',
    ],
  ),
  DemoHospital(
    id: 'bh-mia',
    name: 'Baptist Health Miami (Demo)',
    addressLine1: '8900 N Kendall Dr',
    city: 'Miami',
    state: 'FL',
    zip: '33176',
    emptyIcuRooms: <String>[
      'ICU 10A',
      'ICU 10C',
      'ICU 11B',
      'ICU 12A',
      'ICU 12D',
    ],
  ),
  DemoHospital(
    id: 'bh-orl',
    name: 'Baptist Health Orlando (Demo)',
    addressLine1: '1414 Kuhl Ave',
    city: 'Orlando',
    state: 'FL',
    zip: '32806',
    emptyIcuRooms: <String>[
      'ICU 5A',
      'ICU 5B',
      'ICU 6C',
      'ICU 7A',
      'ICU 8D',
    ],
  ),
  DemoHospital(
    id: 'bh-tpa',
    name: 'Baptist Health Tampa (Demo)',
    addressLine1: '2727 W Dr Martin Luther King Jr Blvd',
    city: 'Tampa',
    state: 'FL',
    zip: '33607',
    emptyIcuRooms: <String>[
      'ICU 2A',
      'ICU 2D',
      'ICU 3B',
      'ICU 4A',
      'ICU 4C',
    ],
  ),
  DemoHospital(
    id: 'bh-tal',
    name: 'Baptist Health Tallahassee (Demo)',
    addressLine1: '1300 Miccosukee Rd',
    city: 'Tallahassee',
    state: 'FL',
    zip: '32308',
    emptyIcuRooms: <String>[
      'ICU 1B',
      'ICU 1D',
      'ICU 2C',
      'ICU 3C',
    ],
  ),
  DemoHospital(
    id: 'bh-nap',
    name: 'Baptist Health Naples (Demo)',
    addressLine1: '3300 Tamiami Trl N',
    city: 'Naples',
    state: 'FL',
    zip: '34103',
    emptyIcuRooms: <String>[
      'ICU 9A',
      'ICU 9B',
      'ICU 9C',
      'ICU 9D',
    ],
  ),
];


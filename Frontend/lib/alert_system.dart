class PatientRecord {
  const PatientRecord({
    required this.rank,
    required this.name,
    required this.id,
    required this.condition,
    required this.roomNumber,
    this.primaryDoctor,
    this.issue,
    this.vitals,
  });

  final int rank;
  final String name;
  final String id;
  final String condition;
  final String roomNumber;
  final String? primaryDoctor;
  final String? issue;
  final String? vitals;
}

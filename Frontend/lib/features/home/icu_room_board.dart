import '../../models/patient_prediction.dart';
import 'demo_hospitals.dart';

/// One ICU bed slot and whether a patient is assigned.
class IcuRoomStatus {
  const IcuRoomStatus({
    required this.room,
    required this.isOccupied,
    this.patientName,
  });

  final String room;
  final bool isOccupied;
  final String? patientName;
}

/// Builds the full ICU bed board for a hospital from patient room assignments.
abstract final class IcuRoomBoard {
  /// Extra vacant beds beyond the seeded cohort (demo UX).
  static const int extraVacantSlots = 2;

  static List<IcuRoomStatus> forPatients(List<PatientRecord> patients) {
    if (patients.isEmpty) {
      return List.generate(
        patientsPerHospital + extraVacantSlots,
        (i) => IcuRoomStatus(
          room: 'CICU ${(i + 1).toString().padLeft(2, '0')}',
          isOccupied: false,
        ),
      );
    }

    final occupied = <String, String>{};
    var prefix = 'CICU ';
    var padWidth = 2;
    var maxNum = 0;

    for (final p in patients) {
      occupied[p.roomNumber] = p.name;
      final parsed = _parseRoom(p.roomNumber);
      if (parsed != null) {
        prefix = parsed.$1;
        padWidth = parsed.$2;
        maxNum = maxNum < parsed.$3 ? parsed.$3 : maxNum;
      }
    }

    final capacity = [
      patientsPerHospital,
      maxNum + extraVacantSlots,
      occupied.length + extraVacantSlots,
    ].reduce((a, b) => a > b ? a : b);

    final statuses = <IcuRoomStatus>[];
    for (var n = 1; n <= capacity; n++) {
      final room = '$prefix${n.toString().padLeft(padWidth, '0')}';
      final name = occupied[room];
      statuses.add(
        IcuRoomStatus(
          room: room,
          isOccupied: name != null,
          patientName: name,
        ),
      );
    }

    // Include any non-sequential room labels present on patients (safety).
    for (final entry in occupied.entries) {
      if (statuses.any((s) => s.room == entry.key)) continue;
      statuses.add(
        IcuRoomStatus(
          room: entry.key,
          isOccupied: true,
          patientName: entry.value,
        ),
      );
    }

    statuses.sort((a, b) {
      final pa = _parseRoom(a.room);
      final pb = _parseRoom(b.room);
      if (pa != null && pb != null) return pa.$3.compareTo(pb.$3);
      return a.room.compareTo(b.room);
    });

    return statuses;
  }

  /// Returns (prefix, padWidth, roomNumber) e.g. ("CICU ", 2, 1).
  static (String, int, int)? _parseRoom(String room) {
    final match = RegExp(r'^(.+?)(\d+)\s*$').firstMatch(room.trim());
    if (match == null) return null;
    final num = int.tryParse(match.group(2)!);
    if (num == null) return null;
    final digits = match.group(2)!;
    return (match.group(1)!, digits.length, num);
  }
}

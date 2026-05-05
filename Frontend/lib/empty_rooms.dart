import 'package:flutter/material.dart';

class EmptyRoomsButton extends StatefulWidget {
  const EmptyRoomsButton({
    super.key,
    required this.emptyRooms,
    this.label = 'Empty rooms',
  });

  final List<String> emptyRooms;
  final String label;

  @override
  State<EmptyRoomsButton> createState() => _EmptyRoomsButtonState();
}

class _EmptyRoomsButtonState extends State<EmptyRoomsButton> {
  final GlobalKey _buttonKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      key: _buttonKey,
      width: 52,
      height: 52,
      child: FilledButton(
        onPressed: () => _showEmptyRooms(context),
        style: FilledButton.styleFrom(
          backgroundColor: const Color(0xFF393E46),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.all(12),
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: Text(widget.label),
      ),
    );
  }

  Future<void> _showEmptyRooms(BuildContext context) async {
    final buttonContext = _buttonKey.currentContext;
    if (buttonContext == null) {
      return;
    }

    if (widget.emptyRooms.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No empty ICU rooms for this hospital.')),
      );
      return;
    }

    final overlay = Overlay.of(context).context.findRenderObject() as RenderBox;
    final buttonBox = buttonContext.findRenderObject() as RenderBox;
    final buttonTopLeft = buttonBox.localToGlobal(Offset.zero, ancestor: overlay);

    final menuHeight = (widget.emptyRooms.length + 1) * 48.0;
    final menuWidth = buttonBox.size.width;
    final left = buttonTopLeft.dx;
    final top = buttonTopLeft.dy - menuHeight - 8;

    await showMenu<void>(
      context: context,
      color: Colors.white,
      position: RelativeRect.fromLTRB(
        left,
        top,
        overlay.size.width - left - menuWidth,
        overlay.size.height - top - menuHeight,
      ),
      items: [
        const PopupMenuItem<void>(
          enabled: false,
          child: Text(
            'Empty ICU rooms',
            style: TextStyle(fontWeight: FontWeight.w700),
          ),
        ),
        ...widget.emptyRooms.map(
          (room) => PopupMenuItem<void>(
            enabled: false,
            child: Text(room),
          ),
        ),
      ],
    );
  }
}
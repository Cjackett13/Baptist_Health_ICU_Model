import 'package:flutter/material.dart';

import 'icu_room_board.dart';

/// Opens the ICU room tracker panel (all beds — occupied or vacant).
class IcuRoomTrackerButton extends StatefulWidget {
  const IcuRoomTrackerButton({
    super.key,
    required this.rooms,
  });

  final List<IcuRoomStatus> rooms;

  @override
  State<IcuRoomTrackerButton> createState() => _IcuRoomTrackerButtonState();
}

/// @deprecated Use [IcuRoomTrackerButton].
typedef EmptyRoomsButton = IcuRoomTrackerButton;

class _IcuRoomTrackerButtonState extends State<IcuRoomTrackerButton> {
  final GlobalKey _buttonKey = GlobalKey();
  bool _dialogOpen = false;

  int get _occupiedCount =>
      widget.rooms.where((r) => r.isOccupied).length;
  int get _vacantCount => widget.rooms.length - _occupiedCount;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return SizedBox(
      key: _buttonKey,
      width: 52,
      height: 52,
      child: FilledButton(
        onPressed: () {
          if (_dialogOpen) {
            Navigator.of(context, rootNavigator: true).pop();
            return;
          }
          _showTracker(context);
        },
        style: FilledButton.styleFrom(
          backgroundColor: scheme.secondary,
          foregroundColor: scheme.onSecondary,
          padding: EdgeInsets.zero,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        child: const Icon(Icons.bed_rounded, size: 24),
      ),
    );
  }

  Future<void> _showTracker(BuildContext context) async {
    final buttonContext = _buttonKey.currentContext;
    if (buttonContext == null) return;

    final scheme = Theme.of(context).colorScheme;

    if (widget.rooms.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'No ICU rooms to display for this hospital.',
            style: TextStyle(color: scheme.onSecondary),
          ),
          backgroundColor: scheme.secondary,
          behavior: SnackBarBehavior.floating,
          margin: const EdgeInsets.all(16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      );
      return;
    }

    final overlayBox =
        Overlay.of(context).context.findRenderObject() as RenderBox?;
    if (overlayBox == null) return;

    final buttonBox = buttonContext.findRenderObject() as RenderBox;
    final buttonTopLeft =
        buttonBox.localToGlobal(Offset.zero, ancestor: overlayBox);

    const double panelWidth = 300;
    const double horizontalMargin = 10;
    const double gap = 10;

    final media = MediaQuery.sizeOf(context);
    final padding = MediaQuery.paddingOf(context);

    double left =
        buttonTopLeft.dx + (buttonBox.size.width - panelWidth) / 2;
    left = left.clamp(
      horizontalMargin + padding.left,
      media.width - panelWidth - horizontalMargin - padding.right,
    );

    const headerBlock = 88.0;
    const rowHeight = 52.0;
    const footerPad = 14.0;
    const maxListHeight = 320.0;
    final listHeight =
        (widget.rooms.length * rowHeight).clamp(0.0, maxListHeight).toDouble();
    final panelHeight = headerBlock + listHeight + footerPad;

    double top = buttonTopLeft.dy - panelHeight - gap;
    if (top < padding.top + gap) {
      top = buttonTopLeft.dy + buttonBox.size.height + gap;
    }
    if (top + panelHeight > media.height - padding.bottom - gap) {
      top = (media.height - padding.bottom - panelHeight - gap)
          .clamp(padding.top + gap, double.infinity);
    }

    if (!context.mounted) return;

    _dialogOpen = true;
    try {
      await showGeneralDialog<void>(
        context: context,
        useRootNavigator: true,
        barrierDismissible: true,
        barrierLabel:
            MaterialLocalizations.of(context).modalBarrierDismissLabel,
        barrierColor: scheme.onSurface.withValues(alpha: 0.45),
        transitionDuration: const Duration(milliseconds: 200),
        transitionBuilder: (ctx, animation, _, child) {
          final curved = CurvedAnimation(
            parent: animation,
            curve: Curves.easeOutCubic,
            reverseCurve: Curves.easeInCubic,
          );
          return FadeTransition(
            opacity: curved,
            child: ScaleTransition(
              scale: Tween<double>(begin: 0.94, end: 1).animate(curved),
              alignment: Alignment.topCenter,
              child: child,
            ),
          );
        },
        pageBuilder: (dialogContext, _, __) {
          final dialogScheme = Theme.of(dialogContext).colorScheme;
          final divider = dialogScheme.onSurface.withValues(alpha: 0.08);
          final muted = dialogScheme.onSurface.withValues(alpha: 0.55);
          const vacantColor = Color(0xFF4A9E6A);
          const occupiedColor = Color(0xFFE05A5A);

          return SafeArea(
            child: SizedBox.expand(
              child: Material(
                color: Colors.transparent,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Positioned.fill(
                      child: GestureDetector(
                        behavior: HitTestBehavior.opaque,
                        onTap: () =>
                            Navigator.of(dialogContext, rootNavigator: true)
                                .pop(),
                      ),
                    ),
                    Positioned(
                      left: left,
                      top: top,
                      width: panelWidth,
                      child: Material(
                        elevation: 18,
                        shadowColor:
                            dialogScheme.onSurface.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(16),
                        color: dialogScheme.surface,
                        clipBehavior: Clip.antiAlias,
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Container(
                              height: 4,
                              color: dialogScheme.primary,
                            ),
                            Padding(
                              padding:
                                  const EdgeInsets.fromLTRB(14, 12, 14, 10),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  DecoratedBox(
                                    decoration: BoxDecoration(
                                      color: dialogScheme.primary
                                          .withValues(alpha: 0.16),
                                      borderRadius: BorderRadius.circular(10),
                                    ),
                                    child: Padding(
                                      padding: const EdgeInsets.all(8),
                                      child: Icon(
                                        Icons.bed_outlined,
                                        size: 20,
                                        color: dialogScheme.primary,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          'ICU room tracker',
                                          style: TextStyle(
                                            fontWeight: FontWeight.w700,
                                            fontSize: 15,
                                            height: 1.2,
                                            color: dialogScheme.onSurface,
                                          ),
                                        ),
                                        const SizedBox(height: 4),
                                        Text(
                                          '$_occupiedCount occupied · '
                                          '$_vacantCount vacant',
                                          style: TextStyle(
                                            fontSize: 12,
                                            height: 1.2,
                                            color: muted,
                                          ),
                                        ),
                                        const SizedBox(height: 6),
                                        Row(
                                          children: [
                                            _LegendDot(
                                              color: occupiedColor,
                                              label: 'Occupied',
                                            ),
                                            const SizedBox(width: 12),
                                            _LegendDot(
                                              color: vacantColor,
                                              label: 'Vacant',
                                            ),
                                          ],
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            Divider(height: 1, thickness: 1, color: divider),
                            ConstrainedBox(
                              constraints:
                                  BoxConstraints(maxHeight: maxListHeight),
                              child: ListView.separated(
                                shrinkWrap: true,
                                padding: const EdgeInsets.fromLTRB(
                                  10,
                                  6,
                                  10,
                                  10,
                                ),
                                physics: widget.rooms.length > 6
                                    ? const BouncingScrollPhysics()
                                    : const NeverScrollableScrollPhysics(),
                                itemCount: widget.rooms.length,
                                separatorBuilder: (_, __) => Divider(
                                  height: 1,
                                  thickness: 1,
                                  color: divider,
                                ),
                                itemBuilder: (ctx, index) {
                                  final status = widget.rooms[index];
                                  final occupied = status.isOccupied;
                                  final statusColor =
                                      occupied ? occupiedColor : vacantColor;
                                  return Padding(
                                    padding: const EdgeInsets.symmetric(
                                      vertical: 8,
                                      horizontal: 4,
                                    ),
                                    child: Row(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Icon(
                                          occupied
                                              ? Icons.person_outline
                                              : Icons.door_sliding_outlined,
                                          size: 18,
                                          color: statusColor,
                                        ),
                                        const SizedBox(width: 10),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                status.room,
                                                maxLines: 1,
                                                overflow: TextOverflow.ellipsis,
                                                style: TextStyle(
                                                  fontSize: 14,
                                                  fontWeight: FontWeight.w700,
                                                  color:
                                                      dialogScheme.onSurface,
                                                ),
                                              ),
                                              const SizedBox(height: 2),
                                              Text(
                                                occupied
                                                    ? status.patientName ??
                                                        'Occupied'
                                                    : 'Vacant — available',
                                                maxLines: 2,
                                                overflow: TextOverflow.ellipsis,
                                                style: TextStyle(
                                                  fontSize: 11,
                                                  height: 1.25,
                                                  color: muted,
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                        Container(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 8,
                                            vertical: 4,
                                          ),
                                          decoration: BoxDecoration(
                                            color: statusColor
                                                .withValues(alpha: 0.12),
                                            borderRadius:
                                                BorderRadius.circular(12),
                                          ),
                                          child: Text(
                                            occupied ? 'Occupied' : 'Vacant',
                                            style: TextStyle(
                                              fontSize: 10,
                                              fontWeight: FontWeight.w700,
                                              color: statusColor,
                                            ),
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
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      );
    } finally {
      if (mounted) _dialogOpen = false;
    }
  }
}

class _LegendDot extends StatelessWidget {
  const _LegendDot({required this.color, required this.label});
  final Color color;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(shape: BoxShape.circle, color: color),
        ),
        const SizedBox(width: 5),
        Text(
          label,
          style: TextStyle(fontSize: 10, color: color.withValues(alpha: 0.9)),
        ),
      ],
    );
  }
}

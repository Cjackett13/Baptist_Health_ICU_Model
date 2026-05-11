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
  bool _emptyRoomsDialogOpen = false;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    return SizedBox(
      key: _buttonKey,
      width: 52,
      height: 52,
      child: FilledButton(
        onPressed: () {
          if (_emptyRoomsDialogOpen) {
            Navigator.of(context, rootNavigator: true).pop();
            return;
          }
          _showEmptyRooms(context);
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

  Future<void> _showEmptyRooms(BuildContext context) async {
    final buttonContext = _buttonKey.currentContext;
    if (buttonContext == null) {
      return;
    }

    final scheme = Theme.of(context).colorScheme;

    if (widget.emptyRooms.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'No empty ICU rooms for this hospital.',
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
    if (overlayBox == null) {
      return;
    }

    final buttonBox = buttonContext.findRenderObject() as RenderBox;
    final buttonTopLeft =
        buttonBox.localToGlobal(Offset.zero, ancestor: overlayBox);

    const double panelWidth = 268;
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

    const headerBlock = 72.0;
    const rowHeight = 46.0;
    const footerPad = 14.0;
    const maxListHeight = 260.0;
    final listHeight = (widget.emptyRooms.length * rowHeight)
        .clamp(0.0, maxListHeight)
        .toDouble();
    final panelHeight = headerBlock + listHeight + footerPad;

    double top = buttonTopLeft.dy - panelHeight - gap;
    if (top < padding.top + gap) {
      top = buttonTopLeft.dy + buttonBox.size.height + gap;
    }
    if (top + panelHeight > media.height - padding.bottom - gap) {
      top = (media.height - padding.bottom - panelHeight - gap)
          .clamp(padding.top + gap, double.infinity);
    }

    if (!context.mounted) {
      return;
    }

    _emptyRoomsDialogOpen = true;
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
                    shadowColor: dialogScheme.onSurface.withValues(alpha: 0.2),
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
                          padding: const EdgeInsets.fromLTRB(14, 12, 14, 10),
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
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Empty ICU rooms',
                                      style: TextStyle(
                                        fontWeight: FontWeight.w700,
                                        fontSize: 15,
                                        height: 1.2,
                                        color: dialogScheme.onSurface,
                                      ),
                                    ),
                                    const SizedBox(height: 2),
                                    Text(
                                      '${widget.emptyRooms.length} available',
                                      style: TextStyle(
                                        fontSize: 12,
                                        height: 1.2,
                                        color: muted,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                        Divider(height: 1, thickness: 1, color: divider),
                        ConstrainedBox(
                          constraints: BoxConstraints(maxHeight: maxListHeight),
                          child: ListView.separated(
                            shrinkWrap: true,
                            padding: const EdgeInsets.fromLTRB(
                              10,
                              6,
                              10,
                              10,
                            ),
                            physics: widget.emptyRooms.length > 5
                                ? const BouncingScrollPhysics()
                                : const NeverScrollableScrollPhysics(),
                            itemCount: widget.emptyRooms.length,
                            separatorBuilder: (_, __) => Divider(
                              height: 1,
                              thickness: 1,
                              color: divider,
                            ),
                            itemBuilder: (ctx, index) {
                              final room = widget.emptyRooms[index];
                              return Padding(
                                padding: const EdgeInsets.symmetric(
                                  vertical: 8,
                                  horizontal: 4,
                                ),
                                child: Row(
                                  children: [
                                    Icon(
                                      Icons.door_sliding_outlined,
                                      size: 18,
                                      color: dialogScheme.primary
                                          .withValues(alpha: 0.9),
                                    ),
                                    const SizedBox(width: 10),
                                    Expanded(
                                      child: Text(
                                        room,
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                        style: TextStyle(
                                          fontSize: 14,
                                          fontWeight: FontWeight.w600,
                                          color: dialogScheme.onSurface,
                                        ),
                                      ),
                                    ),
                                    Container(
                                      width: 8,
                                      height: 8,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        color: dialogScheme.primary
                                            .withValues(alpha: 0.85),
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
      if (mounted) {
        _emptyRoomsDialogOpen = false;
      }
    }
  }
}

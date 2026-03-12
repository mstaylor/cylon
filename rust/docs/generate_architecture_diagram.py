#!/usr/bin/env python3
"""Generate architecture diagram for Unified Checkpointing Design."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_checkpointing_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colors
    app_color = '#E3F2FD'  # Light blue
    manager_color = '#FFF3E0'  # Light orange
    trait_color = '#E8F5E9'  # Light green
    impl_color = '#FCE4EC'  # Light pink
    shared_color = '#F3E5F5'  # Light purple
    border_color = '#333333'

    # Helper function to draw a box with text
    def draw_box(x, y, width, height, label, sublabel=None, color='white', fontsize=10):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(box)

        if sublabel:
            ax.text(x + width/2, y + height/2 + 0.15, label,
                   ha='center', va='center', fontsize=fontsize, fontweight='bold')
            ax.text(x + width/2, y + height/2 - 0.2, sublabel,
                   ha='center', va='center', fontsize=fontsize-2, style='italic')
        else:
            ax.text(x + width/2, y + height/2, label,
                   ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5))

    # Application Code box (top)
    draw_box(1.5, 6.8, 8, 1.2, 'Application Code', color=app_color, fontsize=12)
    #ax.text(5.5, 7.2, 'let mgr = CheckpointManager::new(config);',
     #      ha='center', va='center', fontsize=8, family='monospace')
    #ax.text(5.5, 6.95, 'mgr.checkpoint(&[("table1", &table1)])?;',
    #       ha='center', va='center', fontsize=8, family='monospace')

    # Arrow from Application to Manager
    draw_arrow(5.5, 6.8, 5.5, 6.3)

    # CheckpointManager box
    draw_box(1.5, 4.8, 8, 1.4, 'CheckpointManager', color=manager_color, fontsize=12)
    #ax.text(5.5, 5.35, '• Orchestrates checkpoint lifecycle',
    #       ha='center', va='center', fontsize=8)
    #ax.text(5.5, 5.1, '• Delegates to trait implementations',
    #       ha='center', va='center', fontsize=8)
    #ax.text(5.5, 4.85, '• Environment-agnostic logic',
    #       ha='center', va='center', fontsize=8)

    # Arrows from Manager to Traits
    draw_arrow(3.0, 4.8, 2.5, 4.3)
    draw_arrow(5.5, 4.8, 5.5, 4.3)
    draw_arrow(8.0, 4.8, 8.5, 4.3)

    # Trait boxes (middle layer)
    draw_box(1.2, 3.5, 2.6, 0.8, 'Coordinator', 'Trait', color=trait_color)
    draw_box(4.2, 3.5, 2.6, 0.8, 'Storage', 'Trait', color=trait_color)
    draw_box(7.2, 3.5, 2.6, 0.8, 'Serializer', 'Trait', color=trait_color)

    # Arrows from Traits to Implementations
    draw_arrow(2.0, 3.5, 1.7, 2.9)
    draw_arrow(3.0, 3.5, 3.3, 2.9)
    draw_arrow(5.0, 3.5, 4.7, 2.9)
    draw_arrow(6.0, 3.5, 6.3, 2.9)
    draw_arrow(8.5, 3.5, 8.5, 2.9)

    # Implementation boxes (bottom layer)
    # Coordinator implementations
    draw_box(0.8, 2.0, 1.6, 0.8, 'MPI', 'Coordinator', color=impl_color)
    draw_box(2.6, 2.0, 1.6, 0.8, 'Redis', 'Coordinator', color=impl_color)

    # Storage implementations
    draw_box(4.0, 2.0, 1.4, 0.8, 'S3', 'Storage', color=impl_color)
    draw_box(5.6, 2.0, 1.6, 0.8, 'Lustre', 'Storage', color=impl_color)

    # Serializer implementations
    draw_box(7.4, 2.0, 2.4, 0.8, 'Arrow IPC /', 'Parquet', color=shared_color)

    # Environment labels
    ax.text(1.6, 1.7, 'HPC', ha='center', va='center', fontsize=8, color='#666666')
    ax.text(3.4, 1.7, 'Serverless', ha='center', va='center', fontsize=8, color='#666666')
    ax.text(4.7, 1.7, 'Both', ha='center', va='center', fontsize=8, color='#666666')
    ax.text(6.4, 1.7, 'HPC', ha='center', va='center', fontsize=8, color='#666666')
    ax.text(8.6, 1.7, 'Shared', ha='center', va='center', fontsize=8, color='#666666')

    # Legend
    legend_y = 0.8
    legend_x = 1.0
    box_size = 0.25
    spacing = 2.0

    ax.text(5.5, 1.1, 'Legend', ha='center', va='center', fontsize=10, fontweight='bold')

    items = [
        (app_color, 'Application'),
        (manager_color, 'Manager'),
        (trait_color, 'Traits'),
        (impl_color, 'Implementations'),
        (shared_color, 'Shared'),
    ]

    start_x = 1.5
    for i, (color, label) in enumerate(items):
        x = start_x + i * spacing
        box = FancyBboxPatch((x, legend_y - box_size/2), box_size, box_size,
                             boxstyle="round,pad=0.01,rounding_size=0.05",
                             facecolor=color, edgecolor=border_color, linewidth=1)
        ax.add_patch(box)
        ax.text(x + box_size + 0.1, legend_y, label,
               ha='left', va='center', fontsize=8)

    # Title
    ax.text(5.5, 8.2, 'Unified Checkpointing Architecture',
           ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/parallels/cylon/rust/docs/checkpointing_architecture.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("Diagram saved to /home/parallels/cylon/rust/docs/checkpointing_architecture.png")

if __name__ == '__main__':
    create_checkpointing_architecture()

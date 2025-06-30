import { Component, EventEmitter, Input, OnDestroy, Output } from '@angular/core';

@Component({
  selector: 'app-controls-panel',
  template: `
    <div class="controls-panel">
      <!-- Annotation Mode Selection -->
      <div class="space-y-2">
        <h3 class="text-lg font-medium text-gray-900">Annotation Mode</h3>
        <div class="grid grid-cols-3 gap-2">
          <button
            mat-stroked-button
            [color]="annotationMode === 'text' ? 'primary' : ''"
            (click)="setAnnotationMode('text')"
            matTooltip="Add text labels"
          >
            <!-- <mat-icon>text_fields</mat-icon> -->
            Text
          </button>

          <button
            mat-stroked-button
            [color]="annotationMode === 'click' ? 'primary' : ''"
            (click)="setAnnotationMode('click')"
            matTooltip="Add click annotations (Ctrl+Click for negative)"
          >
            <!-- <mat-icon>touch_app</mat-icon> -->
            Click
          </button>

          <button
            mat-stroked-button
            [color]="annotationMode === 'eraser' ? 'primary' : ''"
            (click)="setAnnotationMode('eraser')"
            matTooltip="Erase annotations"
          >
            <!-- <mat-icon>auto_fix_high</mat-icon> -->
            Eraser
          </button>

          <button
            mat-stroked-button
            [color]="annotationMode === 'polygon' ? 'primary' : ''"
            (click)="setAnnotationMode('polygon')"
            matTooltip="Draw polygon (Ctrl+Click for negative, Enter to complete, Esc to cancel)"
          >
            <!-- <mat-icon>polyline</mat-icon> -->
            Polygon
          </button>
        </div>
      </div>

      <!-- Text Labels -->
      <div class="space-y-2" *ngIf="annotationMode === 'text'">
        <h3 class="text-lg font-medium text-gray-900">Text Labels</h3>
        <mat-form-field appearance="outline" class="w-full">
          <input
            matInput
            [ngModel]="labels.join(', ')"
            (ngModelChange)="updateLabels($event)"
            placeholder="e.g. car, person, tree"
          >
        </mat-form-field>
      </div>

      <!-- Eraser Controls -->
      <div class="space-y-2" *ngIf="annotationMode === 'eraser'">
        <h3 class="text-lg font-medium text-gray-900">Eraser Size</h3>
        <div class="eraser-controls">
          <mat-slider
            class="size-control"
            min="1"
            max="50"
            step="1"
            discrete
          >
            <input
              matSliderThumb
              [value]="eraserSize"
              (valueChange)="updateEraserSize($event)"
            >
          </mat-slider>
          <span class="text-sm text-gray-600">{{ eraserSize }}px</span>
        </div>
      </div>

      <!-- Keyboard Shortcuts -->
      <div class="mt-6 p-4 bg-gray-50 rounded-lg">
        <h4 class="text-sm font-medium text-gray-900 mb-2">Keyboard Shortcuts</h4>
        <ul class="text-sm text-gray-600 space-y-1">
          <li>
            <kbd class="bg-white rounded shadow-sm">Ctrl</kbd>
            + Click: Negative annotation
          </li>
          <li>
            <kbd class="px-2 py-1 bg-white rounded shadow-sm">E</kbd>
            : Switch to eraser
          </li>
          <li>
            <kbd class="px-2 py-1 bg-white rounded shadow-sm">P</kbd>
            : Switch to polygon
          </li>
          <li>
            <kbd class="px-2 py-1 bg-white rounded shadow-sm">Enter</kbd>
            : Complete polygon
          </li>
          <li>
            <kbd class="px-2 py-1 bg-white rounded shadow-sm">Esc</kbd>
            : Cancel polygon
          </li>
        </ul>
      </div>
    </div>
  `,
  styles: [`
    .controls-panel {
      background: white;
      padding: 1rem;
      border-radius: 0.5rem;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
  `]
})
export class ControlsPanelComponent implements OnDestroy {
  @Input() annotationMode: 'text' | 'click' | 'eraser' | 'polygon' = 'click';
  @Output() annotationModeChange = new EventEmitter<'text' | 'click' | 'eraser' | 'polygon'>();
  
  @Input() eraserSize = 20;
  @Output() eraserSizeChange = new EventEmitter<number>();

  @Input() labels: string[] = [];
  @Output() labelsChange = new EventEmitter<string[]>();

  private boundHandleKeydown: (e: KeyboardEvent) => void;

  constructor() {
    this.boundHandleKeydown = this.handleKeydown.bind(this);
  }

  ngOnInit(): void {
    document.addEventListener('keydown', this.boundHandleKeydown);
  }

  setAnnotationMode(mode: 'text' | 'click' | 'eraser' | 'polygon'): void {
    this.annotationMode = mode;
    this.annotationModeChange.emit(mode);
  }

  updateEraserSize(size: number | null): void {
    if (size !== null) {
      this.eraserSize = size;
      this.eraserSizeChange.emit(size);
    }
  }

  updateLabels(value: string): void {
    this.labels = value.split(',').map(label => label.trim()).filter(label => label);
    this.labelsChange.emit(this.labels);
  }

  private handleKeydown(e: KeyboardEvent): void {
    // Only handle shortcuts when not typing in an input
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
      return;
    }

    switch (e.key.toLowerCase()) {
      case 'e':
        this.setAnnotationMode('eraser');
        break;
      case 'p':
        this.setAnnotationMode('polygon');
        break;
    }
  }

  ngOnDestroy(): void {
    document.removeEventListener('keydown', this.boundHandleKeydown);
  }
} 
import { Component, EventEmitter, Input, Output } from '@angular/core';

interface PreprocessingOptions {
  metadata: boolean;
  grayscale: boolean;
  grayscaleThreshold: number;
  binarize: boolean;
  binarizeThreshold: number;
  binarizeInverse: boolean;
  normalize: boolean;
  remove_noise: boolean;
  binarizeGrayscale: boolean;
  resize: boolean;
  resizeHeight: number;
  resizeWidth: number;
  keepAspect: boolean;
  bilinear: boolean;
  bicubic: boolean;
  lanczos: boolean;
}

@Component({
  selector: 'app-preprocessing-options',
  template: `
    <div class="preprocessing-options">
      <h3 class="text-lg font-medium text-gray-900 mb-4">Preprocessing Options</h3>

      <!-- Metadata -->
      <div class="option-group mb-4">
        <mat-checkbox
          [(ngModel)]="options.metadata"
          (ngModelChange)="updateOptions()"
        >
          Extract Metadata
        </mat-checkbox>
      </div>

      <!-- Grayscale -->
      <div class="option-group mb-4">
        <mat-checkbox
          [(ngModel)]="options.grayscale"
          (ngModelChange)="updateOptions()"
        >
          Convert to Grayscale
        </mat-checkbox>

        <div *ngIf="options.grayscale" class="mt-2">
          <div class="flex-row items-center">
            <label class="text-sm text-gray-600 mr-3 min-w-[80px]">Threshold:</label>
            <div class="flex-row items-center">
            <span class="text-sm text-gray-600 ml-3 min-w-[40px] mr-1">0</span>
            <mat-slider
              style="margin-right: 2px;"
              class="flex-grow"
              [min]="0"
              [max]="255"
              [step]="1"
              [discrete]="true"
              [showTickMarks]="true"
            >
              <input 
                matSliderThumb
                [(ngModel)]="options.grayscaleThreshold"
                (ngModelChange)="updateOptions()"
              >
            </mat-slider>
            <span class="text-sm text-gray-600 ml-3 min-w-[40px]">255</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Binarize -->
      <div class="option-group mb-4">
        <mat-checkbox
          [(ngModel)]="options.binarize"
          (ngModelChange)="updateOptions()"
        >
          Binarize Image
        </mat-checkbox>

        <div *ngIf="options.binarize" class="space-y-2 mt-2">
          <div class="flex-row items-center">
            <label class="text-sm text-gray-600 mr-3 min-w-[80px]">Threshold:</label>
            <div class="flex-row items-center">
            <span class="text-sm text-gray-600 ml-3 min-w-[40px] mr-1">0</span>
            <mat-slider
              style="margin-right: 2px;"
              class="flex-grow"
              [min]="0"
              [max]="255"
              [step]="1"
              [discrete]="true"
              [showTickMarks]="true"
            >
              <input 
                matSliderThumb
                [(ngModel)]="options.binarizeThreshold"
                (ngModelChange)="updateOptions()"
              >
            </mat-slider>
            <span class="text-sm text-gray-600 ml-3 min-w-[40px]">255</span>
            </div>
          </div>

          <div class="space-y-2">
            <mat-slide-toggle
              [(ngModel)]="options.binarizeInverse"
              (ngModelChange)="updateOptions()"
            >
              Inverse Colors
            </mat-slide-toggle>

            <mat-slide-toggle
              [(ngModel)]="options.normalize"
              (ngModelChange)="updateOptions()"
            >
              Normalize
            </mat-slide-toggle>

            <mat-slide-toggle
              [(ngModel)]="options.remove_noise"
              (ngModelChange)="updateOptions()"
            >
              Remove Noise
            </mat-slide-toggle>

            <mat-slide-toggle
              [(ngModel)]="options.binarizeGrayscale"
              (ngModelChange)="updateOptions()"
            >
              Convert to Grayscale
            </mat-slide-toggle>
          </div>
        </div>
      </div>

      <!-- Resize -->
      <div class="option-group mb-4">
        <mat-checkbox
          [(ngModel)]="options.resize"
          (ngModelChange)="updateOptions()"
        >
          Resize Image
        </mat-checkbox>

        <div *ngIf="options.resize" class="space-y-2 mt-2">
          <div class="grid grid-cols-2 gap-4">
            <mat-form-field appearance="outline">
              <mat-label>Width</mat-label>
              <input
                matInput
                type="number"
                [(ngModel)]="options.resizeWidth"
                (ngModelChange)="updateOptions()"
              >
            </mat-form-field>

            <mat-form-field appearance="outline">
              <mat-label>Height</mat-label>
              <input
                matInput
                type="number"
                [(ngModel)]="options.resizeHeight"
                (ngModelChange)="updateOptions()"
              >
            </mat-form-field>
          </div>

          <mat-slide-toggle
            [(ngModel)]="options.keepAspect"
            (ngModelChange)="updateOptions()"
          >
            Keep Aspect Ratio
          </mat-slide-toggle>

          <!-- Interpolation Options -->
          <div style="margin-top: 25px;">
            <label class="text-sm font-medium text-gray-900 mb-2 block">Interpolation:</label>
            <div class="space-y-2 ml-4">
              <mat-checkbox
                [(ngModel)]="options.bilinear"
                (ngModelChange)="updateOptions()"
              >
                Bilinear
              </mat-checkbox>

              <mat-checkbox
                [(ngModel)]="options.bicubic"
                (ngModelChange)="updateOptions()"
              >
                Bicubic
              </mat-checkbox>

              <mat-checkbox
                [(ngModel)]="options.lanczos"
                (ngModelChange)="updateOptions()"
              >
                Lanczos
              </mat-checkbox>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .preprocessing-options {
      padding: 1rem;
      background: white;
      border-radius: 0.5rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    mat-slider {
      min-width: 150px;
    }
    
    .option-group {
      border-bottom: 1px solid #eee;
      padding-bottom: 1rem;
    }

    mat-checkbox {
      display: block;
      margin-bottom: 0.5rem;
    }
  `]
})
export class PreprocessingOptionsComponent {
  @Input() options: PreprocessingOptions = {
    metadata: false,
    grayscale: false,
    grayscaleThreshold: 128,
    binarize: false,
    binarizeThreshold: 128,
    binarizeInverse: false,
    normalize: false,
    remove_noise: false,
    binarizeGrayscale: false,
    resize: false,
    resizeHeight: 0,
    resizeWidth: 0,
    keepAspect: true,
    bilinear: false,
    bicubic: false,
    lanczos: false
  };

  @Output() optionsChange = new EventEmitter<PreprocessingOptions>();

  updateOptions(): void {
    this.optionsChange.emit(this.options);
  }

  getInterpolationString(): string {
    const interpolations = [];
    if (this.options.bilinear) interpolations.push('bilinear');
    if (this.options.bicubic) interpolations.push('bicubic');
    if (this.options.lanczos) interpolations.push('lanczos');
    return interpolations.join(',');
  }
} 
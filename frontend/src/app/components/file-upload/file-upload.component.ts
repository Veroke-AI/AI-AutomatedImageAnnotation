import { Component, EventEmitter, Output } from '@angular/core';

@Component({
  selector: 'app-file-upload',
  template: `
    <div class="upload-container p-4 bg-white rounded-lg shadow-sm">
      <h3 class="text-lg font-medium mb-2">Upload Images</h3>
      
      <!-- File Input -->
      <div
        class="upload-area p-4 border-2 border-dashed border-gray-300 rounded-lg text-center cursor-pointer hover:border-primary-500 transition-colors"
        (dragover)="onDragOver($event)"
        (dragleave)="onDragLeave($event)"
        (drop)="onDrop($event)"
        [class.drag-over]="isDragging"
      >
        <input
          type="file"
          #fileInput
          multiple
          accept="image/*"
          class="hidden"
          (change)="onFileSelected($event)"
        >
        
        <mat-icon class="text-4xl mb-2 text-gray-400">cloud_upload</mat-icon>
        <p class="text-gray-600">
          Drag and drop images here or
          <button
            mat-button
            color="primary"
            type="button"
            (click)="fileInput.click()"
          >
            browse
          </button>
        </p>
        <p class="text-sm text-gray-500 mt-1">
          Supported formats: PNG, JPG, JPEG
        </p>
      </div>

      <!-- Upload List -->
      <div class="mt-4 space-y-2" *ngIf="uploadedFiles.length > 0">
        <div
          *ngFor="let file of uploadedFiles"
          class="flex items-center justify-between p-2 bg-gray-50 rounded"
        >
          <span class="text-sm text-gray-600 truncate">{{ file.name }}</span>
          <button
            mat-icon-button
            color="warn"
            (click)="removeFile(file)"
            matTooltip="Remove file"
          >
            <mat-icon>close</mat-icon>
          </button>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .upload-area {
      min-height: 200px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }

    .upload-area.drag-over {
      background-color: rgba(79, 70, 229, 0.05);
      border-color: #4f46e5;
    }

    .hidden {
      display: none;
    }
  `]
})
export class FileUploadComponent {
  @Output() filesSelected = new EventEmitter<File[]>();

  uploadedFiles: File[] = [];
  isDragging = false;

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;

    const files = Array.from(event.dataTransfer?.files || [])
      .filter(file => file.type.startsWith('image/'));
    
    this.addFiles(files);
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      const files = Array.from(input.files)
        .filter(file => file.type.startsWith('image/'));
      this.addFiles(files);
    }
  }

  removeFile(file: File): void {
    const index = this.uploadedFiles.indexOf(file);
    if (index > -1) {
      this.uploadedFiles.splice(index, 1);
      this.filesSelected.emit(this.uploadedFiles);
    }
  }

  private addFiles(files: File[]): void {
    // Add only new files
    const newFiles = files.filter(file => 
      !this.uploadedFiles.some(existing => 
        existing.name === file.name && existing.size === file.size
      )
    );

    this.uploadedFiles = [...this.uploadedFiles, ...newFiles];
    this.filesSelected.emit(this.uploadedFiles);
  }
} 
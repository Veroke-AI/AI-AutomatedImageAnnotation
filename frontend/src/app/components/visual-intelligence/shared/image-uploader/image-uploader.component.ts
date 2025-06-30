import { Component, EventEmitter, Output, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar } from '@angular/material/snack-bar';

@Component({
  selector: 'app-image-uploader',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule
  ],
  template: `
    <div 
      class="border-2 border-dashed border-gray-300 rounded-lg p-6"
      [class.border-blue-500]="isDragging"
      (dragover)="onDragOver($event)"
      (dragleave)="onDragLeave($event)"
      (drop)="onDrop($event)"
    >
      <div class="text-center">
        <div *ngIf="!hasFiles" class="space-y-4">
          <mat-icon class="text-gray-400" style="font-size: 48px; width: 48px; height: 48px; margin: 0 auto;">
            folder_open
          </mat-icon>
          <div class="text-gray-600">
            <p class="text-lg font-medium">Drop a folder with images here</p>
            <p class="text-sm">or</p>
          </div>
          <div>
            <input 
              #fileInputRef
              type="file"
              webkitdirectory
              multiple
              class="hidden"
              (change)="onFileSelected($event)"
            >
            <button 
              mat-raised-button
              color="primary"
              (click)="openFileInput()"
            >
              Select Folder
            </button>
          </div>
        </div>

        <div *ngIf="hasFiles" class="space-y-4">
          <div class="flex items-center justify-center gap-2">
            <mat-icon class="text-green-500">check_circle</mat-icon>
            <span class="text-gray-900">{{ uploadedFiles.length }} images found</span>
          </div>
          
          <!-- File List -->
          <div class="max-h-48 overflow-y-auto">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div *ngFor="let file of uploadedFiles" class="relative group">
                <img 
                  [src]="getPreviewUrl(file)"
                  class="w-full aspect-square object-cover rounded-lg shadow-sm"
                  [alt]="file.name"
                >
                <button
                  mat-icon-button
                  color="warn"
                  class="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  (click)="removeFile(file)"
                >
                  <mat-icon>close</mat-icon>
                </button>
              </div>
            </div>
          </div>

          <div class="flex justify-center gap-4">
            <button 
              mat-stroked-button
              color="primary"
              (click)="openFileInput()"
            >
              Select Another Folder
            </button>
            <button 
              mat-stroked-button
              color="warn"
              (click)="clearFiles()"
            >
              Clear All
            </button>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
    }
  `]
})
export class ImageUploaderComponent {
  @Output() imagesUploaded = new EventEmitter<File[]>();
  @ViewChild('fileInputRef') fileInputRef!: ElementRef<HTMLInputElement>;

  uploadedFiles: File[] = [];
  isDragging = false;
  previewUrls = new Map<File, string>();

  constructor(private snackBar: MatSnackBar) {}

  get hasFiles(): boolean {
    return this.uploadedFiles.length > 0;
  }

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

    // Handle dropped folder
    const items = Array.from(event.dataTransfer?.items || []);
    if (items.length > 0) {
      const entry = items[0].webkitGetAsEntry();
      if (entry?.isDirectory) {
        this.processDirectory(entry);
      } else {
        const files = Array.from(event.dataTransfer?.files || []);
        this.handleFiles(files);
      }
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      const files = Array.from(input.files);
      this.handleFiles(files);
    }
  }

  async processDirectory(entry: any): Promise<void> {
    const files: File[] = [];
    const processEntry = async (entry: any) => {
      if (entry.isFile) {
        return new Promise<void>((resolve) => {
          entry.file((file: File) => {
            if (file.type.startsWith('image/')) {
              files.push(file);
            }
            resolve();
          });
        });
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        return new Promise<void>((resolve) => {
          reader.readEntries(async (entries: any[]) => {
            await Promise.all(entries.map(processEntry));
            resolve();
          });
        });
      }
    };

    await processEntry(entry);
    this.handleFiles(files);
  }

  handleFiles(files: File[]): void {
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length !== files.length) {
      this.snackBar.open('Some files were skipped because they are not images', 'Close', {
        duration: 3000
      });
    }

    if (imageFiles.length === 0) {
      return;
    }

    // Create preview URLs for new files
    imageFiles.forEach(file => {
      if (!this.previewUrls.has(file)) {
        this.previewUrls.set(file, URL.createObjectURL(file));
      }
    });

    // Replace existing files instead of adding to them
    this.uploadedFiles = imageFiles;
    this.imagesUploaded.emit(this.uploadedFiles);
  }

  removeFile(file: File): void {
    const url = this.previewUrls.get(file);
    if (url) {
      URL.revokeObjectURL(url);
      this.previewUrls.delete(file);
    }

    this.uploadedFiles = this.uploadedFiles.filter(f => f !== file);
    this.imagesUploaded.emit(this.uploadedFiles);
  }

  clearFiles(): void {
    // Revoke all preview URLs
    this.previewUrls.forEach(url => URL.revokeObjectURL(url));
    this.previewUrls.clear();

    this.uploadedFiles = [];
    this.imagesUploaded.emit(this.uploadedFiles);
  }

  getPreviewUrl(file: File): string {
    return this.previewUrls.get(file) || '';
  }

  ngOnDestroy(): void {
    // Clean up preview URLs
    this.previewUrls.forEach(url => URL.revokeObjectURL(url));
  }

  openFileInput(): void {
    this.fileInputRef.nativeElement.click();
  }
} 
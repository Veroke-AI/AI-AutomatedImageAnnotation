import { Component, EventEmitter, Input, OnDestroy, SimpleChanges, OnChanges, Output } from '@angular/core';

@Component({
  selector: 'app-image-gallery',
  template: `
    <div class="space-y-4" style="margin-top:20px">
      <h3 class="text-lg font-medium text-gray-900">Uploaded Images</h3>
      
      <!-- Uploaded Images -->
      <div class="image-gallery">
        <div
          *ngFor="let image of images"
          class="image-item"
          [class.ring-2]="image === selectedImage"
          [class.ring-primary-500]="image === selectedImage"
          (click)="onImageSelect(image)"
        >
          <img
            [src]="imageUrls.get(image.name) || ''"
            [alt]="image.name"
            class="w-full h-full object-cover"
          />
          <!--<div class="absolute bottom-0 left-0 right-0 p-2 bg-black bg-opacity-50">
            <p class="text-xs text-white truncate">{{ image.name }}</p>
          </div>-->
        </div>
      </div>

      <!-- Empty State -->
      <div
        *ngIf="images.length === 0"
        class="text-center py-8 text-gray-500"
      >
        <mat-icon class="text-4xl mb-2" style="height: 35px; width: 35px;">photo_library</mat-icon>
        <p>No images uploaded yet</p>
      </div>
    </div>
  `
})
export class ImageGalleryComponent implements OnChanges, OnDestroy {
  @Input() images: File[] = [];
  @Input() selectedImage: File | null = null;
  @Output() imageSelected = new EventEmitter<File>();

  imageUrls = new Map<string, string>();

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['images']) {
      // Clean up old URLs
      this.cleanupImageUrls();
      
      // Create new URLs
      this.images.forEach(image => {
        if (!this.imageUrls.has(image.name)) {
          this.imageUrls.set(image.name, URL.createObjectURL(image));
        }
      });
    }
  }

  onImageSelect(image: File): void {
    this.imageSelected.emit(image);
  }

  private cleanupImageUrls(): void {
    // Revoke URLs for images that are no longer in the list
    const currentImageNames = new Set(this.images.map(img => img.name));
    for (const [imageName, url] of this.imageUrls.entries()) {
      if (!currentImageNames.has(imageName)) {
        URL.revokeObjectURL(url);
        this.imageUrls.delete(imageName);
      }
    }
  }

  ngOnDestroy(): void {
    // Clean up all URLs
    for (const url of this.imageUrls.values()) {
      URL.revokeObjectURL(url);
    }
    this.imageUrls.clear();
  }
} 
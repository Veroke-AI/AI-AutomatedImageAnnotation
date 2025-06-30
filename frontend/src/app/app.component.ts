import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { CommonModule } from '@angular/common';
import { AnnotationService } from './services/annotation.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatSnackBarModule
  ],
  template: `
    <div class="h-full">
      <router-outlet></router-outlet>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100vh;
    }
  `]
})
export class AppComponent {
  uploadedFiles: File[] = [];
  selectedImage: File | null = null;
  currentAnnotationMode: 'text' | 'click' | 'eraser' | 'polygon' = 'click';
  eraserSize = 20;
  labels: string[] = [];
  preprocessingOptions = {
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

  constructor(
    private snackBar: MatSnackBar,
    private annotationService: AnnotationService
  ) {}

  get canSubmit(): boolean {
    return this.selectedImage !== null && 
           (this.labels.length > 0 || this.annotationService.getAnnotations(this.getImageId(this.selectedImage)).clickAnnotations.length > 0);
  }

  onFilesSelected(files: File[]): void {
    this.uploadedFiles = files;
    if (files.length > 0) {
      this.selectedImage = files[0];
    }
  }

  onImageSelected(image: File): void {
    if (image !== this.selectedImage) {
      this.selectedImage = image;
    }
  }

  onModeChanged(mode: 'text' | 'click' | 'eraser' | 'polygon'): void {
    this.currentAnnotationMode = mode;
  }

  onEraserSizeChanged(size: number): void {
    this.eraserSize = size;
  }

  onAnnotationAdded(event: any): void {
    console.log('Annotation added:', event);
  }

  onAnnotationRemoved(event: any): void {
    console.log('Annotation removed:', event);
  }

  submitAnnotations(): void {
    if (!this.selectedImage) return;

    const formData = new FormData();

    // Add all images from the folder
    this.uploadedFiles.forEach((file, index) => {
      formData.append('images[]', file, file.name);
    });

    // Get annotations for the current image
    const imageId = this.getImageId(this.selectedImage);
    const annotations = this.annotationService.getAnnotations(imageId);

    // Add click annotations data
    // const positiveClicks = annotations.clickAnnotations
    //   .filter(click => click.type === 'positive')
    //   .map(click => ({ x: Math.round(click.x), y: Math.round(click.y) }));
    
    // const negativeClicks = annotations.clickAnnotations
    //   .filter(click => click.type === 'negative')
    //   .map(click => ({ x: Math.round(click.x), y: Math.round(click.y) }));

    // Process polygon data - only include all polygons regardless of type
    const polygons = annotations.polygons.map(polygon => ({
      points: polygon.points.map(point => ({
        x: Math.round(point.x),
        y: Math.round(point.y)
      }))
    }));

    // Create the annotation data object
    const annotationData = {
      imageId: imageId,
      imageName: this.selectedImage.name,
      // positiveClicks,
      // negativeClicks,
      polygons
    };

    // Add annotation data as JSON
    formData.append('annotation_data', JSON.stringify(annotationData));

    // Add only main preprocessing options to actions
    const actions: string[] = [];
    if (this.preprocessingOptions.metadata) actions.push('metadata');
    if (this.preprocessingOptions.grayscale) actions.push('grayscale');
    if (this.preprocessingOptions.binarize) actions.push('binarize');
    if (this.preprocessingOptions.resize) actions.push('resize');

    // Create preprocessing data with proper types
    formData.append('actions', actions.join(','));
    formData.append('grayscale_threshold', Math.round(this.preprocessingOptions.grayscaleThreshold).toString());
    formData.append('binarize_threshold', Math.round(this.preprocessingOptions.binarizeThreshold).toString());
    formData.append('binarize_inverse', this.preprocessingOptions.binarizeInverse ? '1' : '0');
    formData.append('resize_width', Math.round(this.preprocessingOptions.resizeWidth).toString());
    formData.append('resize_height', Math.round(this.preprocessingOptions.resizeHeight).toString());
    formData.append('keep_aspect', this.preprocessingOptions.keepAspect ? '1' : '0');

    // Add binarize options as separate fields
    formData.append('normalize', this.preprocessingOptions.normalize ? '1' : '0');
    formData.append('remove_noise', this.preprocessingOptions.remove_noise ? '1' : '0');
    formData.append('binarize_grayscale', this.preprocessingOptions.binarizeGrayscale ? '1' : '0');

    // Add interpolation options
    const interpolations = [];
    if (this.preprocessingOptions.bilinear) interpolations.push('bilinear');
    if (this.preprocessingOptions.bicubic) interpolations.push('bicubic');
    if (this.preprocessingOptions.lanczos) interpolations.push('lanczos');
    formData.append('interpolation', interpolations.join(','));

    // Log FormData contents
    console.log('Form Data Contents:');
    formData.forEach((value, key) => {
      console.log(`${key}: ${value}`);
    });

    // For now, just show success message
    this.snackBar.open('Annotations submitted successfully!', 'Close', {
      duration: 3000
    });
  }

  private getPreprocessingString(): string {
    const actions: string[] = [];
    
    if (this.preprocessingOptions.metadata) actions.push('metadata');
    if (this.preprocessingOptions.grayscale) actions.push('grayscale');
    if (this.preprocessingOptions.binarize) {
      actions.push('binarize');
      if (this.preprocessingOptions.normalize) actions.push('normalize');
      if (this.preprocessingOptions.remove_noise) actions.push('remove_noise');
      if (this.preprocessingOptions.binarizeGrayscale) actions.push('binarize_grayscale');
    }
    if (this.preprocessingOptions.resize) {
      actions.push('resize');
      const interpolations = [];
      if (this.preprocessingOptions.bilinear) interpolations.push('bilinear');
      if (this.preprocessingOptions.bicubic) interpolations.push('bicubic');
      if (this.preprocessingOptions.lanczos) interpolations.push('lanczos');
      if (interpolations.length > 0) {
        actions.push(`interpolation=${interpolations.join(',')}`);
      }
    }
    
    return actions.join(',');
  }

  private getImageId(image: File): string {
    return `${image.name}-${image.lastModified}`;
  }
} 
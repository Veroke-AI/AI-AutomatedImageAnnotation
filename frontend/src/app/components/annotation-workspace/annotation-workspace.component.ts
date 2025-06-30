import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { FormsModule } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { AnnotationService } from '../../services/annotation.service';
import { CocoDatasetService, CocoCategory, CocoAnnotation, CocoImage } from '../../services/coco-dataset.service';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import * as JSZip from 'jszip';
import { ClassListComponent } from '../class-list/class-list.component';
import { PolygonDrawerComponent } from '../polygon-drawer/polygon-drawer.component';
import { ImageUploaderComponent } from '../image-uploader/image-uploader.component';
import { ImageCanvasComponent } from '../image-canvas/image-canvas.component';
import { ClassSelectionDialogComponent } from '../class-selection-dialog/class-selection-dialog.component';
import { config } from '../../../../src/config';
@Component({
  selector: 'app-annotation-workspace',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatInputModule,
    MatFormFieldModule,
    FormsModule,
    ClassListComponent,
    PolygonDrawerComponent,
    ImageUploaderComponent,
    ImageCanvasComponent,
    ClassSelectionDialogComponent
  ],
  template: `
    <div class="min-h-screen bg-gray-100">
      <!-- Header -->
      <header class="bg-white shadow-sm">
        <div class="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 class="text-2xl font-semibold text-gray-900" style="margin: 0;">Image Annotation</h1>
          <div class="flex gap-4">
            <button 
              mat-raised-button 
              color="primary"
              [disabled]="!canDownload"
              (click)="downloadAnnotations()"
            >
              <mat-icon>download</mat-icon>
              Download Annotations
            </button>
            <button 
              mat-raised-button 
              color="primary"
              (click)="goBack()"
            >
              <mat-icon>arrow_back</mat-icon>
              Back to Home
            </button>
          </div>
        </div>
      </header>

      <main class="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Left sidebar -->
          <div class="lg:col-span-3 space-y-6">
            <!-- Image Upload -->
            <div class="bg-white rounded-lg shadow p-4">
              <h3 class="text-lg font-medium text-gray-900 mb-4">Upload Images</h3>
              <app-image-uploader
                (imagesUploaded)="onFilesSelected($event)"
              ></app-image-uploader>
            </div>

            <!-- Image Selection -->
            <div class="bg-white rounded-lg shadow p-4" *ngIf="uploadedFiles.length > 0">
              <h3 class="text-lg font-medium text-gray-900 mb-4">Select Image</h3>
              <div class="space-y-2 max-h-60 overflow-y-auto">
                <div 
                  *ngFor="let file of uploadedFiles; let i = index"
                  class="p-2 rounded cursor-pointer"
                  [class.bg-blue-100]="currentImageIndex === i"
                  (click)="selectImage(i)"
                >
                  {{ file.name }}
                </div>
              </div>
            </div>

            <!-- Class List -->
            <div class="bg-white rounded-lg shadow p-4" *ngIf="categories.length > 0">
              <h3 class="text-lg font-medium text-gray-900 mb-4"></h3>
              <app-class-list
                [categories]="categories"
                (classSelected)="onClassSelected($event)"
              ></app-class-list>
            </div>
          </div>

          <!-- Main content area -->
          <div class="lg:col-span-9">
            <div class="bg-white rounded-lg shadow p-4 relative flex flex-col">
              <!-- Classes Input and Submit Button -->
              <div class="mb-4 flex items-center gap-4" style="min-width: 0;">
                <mat-form-field appearance="fill" style="display: block; flex: 2;" floatLabel="always">
                  <mat-label>Enter object classes (comma-separated)</mat-label>
                  <input 
                    matInput 
                    [(ngModel)]="classes"
                    (ngModelChange)="onClassesChanged($event)"
                    placeholder="e.g. person, car, tree"
                  >
                </mat-form-field>
                <mat-form-field appearance="fill" style="display: block; flex: 1;" floatLabel="always">
                  <mat-label>Split Value (0-1)</mat-label>
                  <input 
                    matInput 
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    [(ngModel)]="splitValue"
                    (ngModelChange)="validateSplitValue($event)"
                  >
                  <mat-error *ngIf="splitValue < 0 || splitValue > 1">
                    Value must be between 0 and 1
                  </mat-error>
                </mat-form-field>
                <button
                  mat-raised-button
                  color="primary"
                  [disabled]="!canSubmit"
                  (click)="submitAnnotations()"
                  style="height: 56px; margin-bottom: 22px;"
                >
                  Submit Annotations
                </button>
              </div>

              <div class="flex relative">
                <!-- Canvas Area -->
                <div class="flex-1 relative min-h-[400px] flex items-center justify-center">
                  <ng-container *ngIf="(selectedImage || uploadedFiles.length > 0); else placeholder">
                    <ng-container [ngSwitch]="currentAnnotationMode">
                      <!-- Click Mode -->
                      <app-image-canvas *ngSwitchCase="'click'" style="width: 100%; height: 100%;"
                        [selectedImage]="selectedImage || uploadedFiles[0]"
                        [annotationMode]="currentAnnotationMode"
                        [annotations]="currentAnnotations"
                        [isLoading]="isLoading"
                        (loadingChange)="isLoading = $event"
                        (annotationAdded)="onAnnotationAdded($event)"
                      ></app-image-canvas>
                      
                      <!-- Polygon Mode -->
                      <app-polygon-drawer
                        *ngIf="selectedImage && currentAnnotationMode === 'polygon'"
                        [imageFile]="selectedImage"
                        [annotations]="currentAnnotations"
                        [selectedCategoryId]="selectedCategory?.id || null"
                        [annotationMode]="currentAnnotationMode"
                        (boundingBoxDrawn)="onBoundingBoxDrawn($event)"
                        (boundingBoxRemoved)="onBoundingBoxRemoved($event)"
                      ></app-polygon-drawer>
                    </ng-container>
                  </ng-container>
                  <ng-template #placeholder>
                    <div class="flex flex-col items-center justify-center w-full h-full min-h-[400px]" style="overflow: visible;">
                      <mat-icon style="font-size: 64px; width: 64px; height: 64px; color: #cbd5e1;">folder_open</mat-icon>
                      <div class="mt-4 text-lg text-gray-500 font-medium text-center">Upload a folder of images to start annotating</div>
                    </div>
                  </ng-template>
                </div>

                <!-- Annotation Mode Panel -->
                <div *ngIf="categories.length > 0" class="flex flex-col items-center justify-center ml-6 space-y-4 min-w-[120px]">
                <button 
                    mat-raised-button 
                    [color]="currentAnnotationMode === 'polygon' ? 'primary' : 'basic'"
                    (click)="setAnnotationMode('polygon')"
                    class="w-full"
                  >
                    <mat-icon>timeline</mat-icon>
                    Polygon
                  </button>
                  <button 
                    mat-raised-button 
                    [color]="currentAnnotationMode === 'click' ? 'primary' : 'basic'"
                    (click)="setAnnotationMode('click')"
                    class="w-full"
                  >
                    <mat-icon>mouse</mat-icon>
                    Click
                  </button>
                  
                </div>

                <!-- Navigation Controls -->
                <div class="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-4">
                  <button 
                    mat-fab 
                    color="primary"
                    [disabled]="!canNavigatePrevious"
                    (click)="previousImage()"
                  >
                    <mat-icon>arrow_back</mat-icon>
                  </button>
                  <button 
                    mat-fab 
                    color="primary"
                    [disabled]="!canNavigateNext"
                    (click)="nextImage()"
                  >
                    <mat-icon>arrow_forward</mat-icon>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <!-- Modern Loader -->
      <div *ngIf="isLoading" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white rounded-lg p-8 flex flex-col items-center">
          <div class="loader-ring">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
          </div>
          <p class="mt-4 text-gray-700 font-medium">Processing Images...</p>
        </div>
      </div>

      <!-- Results Gallery -->
      <div #gallerySection *ngIf="resultImages.length > 0" class="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div class="bg-white rounded-lg shadow p-6">
          <h2 class="text-xl font-semibold mb-4">Results Gallery</h2>
          <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            <div *ngFor="let image of resultImages" class="aspect-square relative group">
              <img [src]="image" class="w-full h-full object-cover rounded-lg" 
                   [alt]="'Result image'" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Gallery Modal -->
    <div *ngIf="showGallery" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div class="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xl font-semibold">Results Gallery</h2>
          <button mat-icon-button (click)="closeGallery()">
            <mat-icon>close</mat-icon>
          </button>
        </div>
        
        <div class="grid grid-cols-3 gap-4">
          <div *ngFor="let image of resultImages" class="relative group">
            <img [src]="image" class="w-full h-48 object-cover rounded" />
            <div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300"></div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
    }

    .loader-ring {
      display: inline-block;
      position: relative;
      width: 80px;
      height: 80px;
    }
    .loader-ring div {
      box-sizing: border-box;
      display: block;
      position: absolute;
      width: 64px;
      height: 64px;
      margin: 8px;
      border: 8px solid #3B82F6;
      border-radius: 50%;
      animation: loader-ring 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
      border-color: #3B82F6 transparent transparent transparent;
    }
    .loader-ring div:nth-child(1) {
      animation-delay: -0.45s;
    }
    .loader-ring div:nth-child(2) {
      animation-delay: -0.3s;
    }
    .loader-ring div:nth-child(3) {
      animation-delay: -0.15s;
    }
    @keyframes loader-ring {
      0% {
        transform: rotate(0deg);
      }
      100% {
        transform: rotate(360deg);
      }
    }
  `]
})
export class AnnotationWorkspaceComponent implements OnInit {
  uploadedFiles: File[] = [];
  selectedImage: File | null = null;
  currentImageIndex = 0;
  classes = '';
  splitValue = 0;
  currentAnnotationMode: 'click' | 'polygon' = 'click';
  isProcessing = false;
  showGallery = false;
  resultImages: string[] = [];
  @ViewChild('gallerySection') gallerySection!: ElementRef;
  isLoading = false;
  categories: CocoCategory[] = [];
  selectedCategory: CocoCategory | null = null;
  currentAnnotations: CocoAnnotation[] = [];
  private currentCocoImageId: number | null = null;

  constructor(
    private router: Router,
    private http: HttpClient,
    private snackBar: MatSnackBar,
    private annotationService: AnnotationService,
    private cocoDatasetService: CocoDatasetService,
    private dialog: MatDialog
  ) {}

  ngOnInit(): void {
    // Subscribe to dataset changes
    this.cocoDatasetService.dataset$.subscribe(dataset => {
      if (dataset) {
        this.categories = dataset.categories;
        // Find the current image in the COCO dataset
        if (this.selectedImage) {
          const imageName = this.selectedImage.name;
          const cocoImage = dataset.images.find(img => img.file_name === imageName);
          if (cocoImage) {
            this.currentCocoImageId = cocoImage.id;
            this.updateAnnotationsForCurrentImage();
          }
        }
      }
    });
  }

  get canNavigatePrevious(): boolean {
    return this.currentImageIndex > 0;
  }

  get canNavigateNext(): boolean {
    return this.currentImageIndex < this.uploadedFiles.length - 1;
  }

  get canSubmit(): boolean {
    return this.uploadedFiles.length > 0 && 
           this.classes.trim().length > 0 && 
           this.splitValue >= 0 && 
           this.splitValue <= 1;
  }

  onFilesSelected(files: File[]): void {
    this.uploadedFiles = files;
    this.currentImageIndex = 0;
    this.selectedImage = files.length > 0 ? files[0] : null;
  }

  selectImage(index: number): void {
    this.currentImageIndex = index;
    this.selectedImage = this.uploadedFiles[index];
    
    // Get the COCO image ID from the service based on the filename
    if (this.selectedImage) {
      this.currentCocoImageId = this.cocoDatasetService.getImageIdByFilename(this.selectedImage.name);
      console.log('Set currentCocoImageId:', {
        filename: this.selectedImage.name,
        cocoImageId: this.currentCocoImageId
      });
      this.updateAnnotationsForCurrentImage();
    }
  }

  setAnnotationMode(mode: 'click' | 'polygon'): void {
    this.currentAnnotationMode = mode;
  }

  onClassesChanged(classes: string): void {
    const classList = classes.split(',').map(c => c.trim()).filter(c => c.length > 0);
    this.annotationService.setClasses(classList);
  }

  previousImage(): void {
    if (this.canNavigatePrevious) {
      this.currentImageIndex--;
      this.selectedImage = this.uploadedFiles[this.currentImageIndex];
      // Get the COCO image ID and update annotations
      this.currentCocoImageId = this.cocoDatasetService.getImageIdByFilename(this.selectedImage.name);
      this.updateAnnotationsForCurrentImage();
    }
  }

  nextImage(): void {
    if (this.canNavigateNext) {
      this.currentImageIndex++;
      this.selectedImage = this.uploadedFiles[this.currentImageIndex];
      // Get the COCO image ID and update annotations
      this.currentCocoImageId = this.cocoDatasetService.getImageIdByFilename(this.selectedImage.name);
      this.updateAnnotationsForCurrentImage();
    }
  }

  async onAnnotationAdded(event: any): Promise<void> {
    if (this.isProcessing) return;
    
    this.isProcessing = true;
    this.isLoading = true;
    try {
      // Update local state
      const imageId = this.getImageId(this.selectedImage!);
      this.annotationService.updateAnnotatedImage(imageId, '');
      
      this.snackBar.open('Annotation added successfully!', 'Close', {
        duration: 3000
      });
    } catch (error) {
      this.snackBar.open('Error adding annotation', 'Close', {
        duration: 3000
      });
      console.error('Annotation error:', error);
    } finally {
      this.isProcessing = false;
      this.isLoading = false;
    }
  }

  async submitAnnotations(): Promise<void> {
    if (!this.canSubmit) return;

    const formData = new FormData();
    
    this.uploadedFiles.forEach(file => {
      formData.append('images', file);
    });

    formData.append('classes', this.classes);
    formData.append('split', this.splitValue.toString());

    try {
      this.isLoading = true;
      const response = await this.http.post(`http://${config.IP}:${config.PORT}/annotate`, formData, {
        responseType: 'blob'
      }).toPromise();

      if (response) {
        // Clear previous results
        this.clearResultImages();
        
        // Process the ZIP file
        const zip = new JSZip();
        const loadedZip = await zip.loadAsync(response);
        
        // Extract and process coco_dataset.json
        const cocoDatasetFile = loadedZip.file('coco_dataset.json');
        if (cocoDatasetFile) {
          const cocoDatasetJson = await cocoDatasetFile.async('string');
          const cocoDataset = JSON.parse(cocoDatasetJson);
          
          // Set the dataset in the service
          this.cocoDatasetService.setDataset(cocoDataset);
          
          // Update local categories
          this.categories = cocoDataset.categories;
          console.log('Updated categories:', this.categories);

          // Set annotation mode to polygon after successful submission
          this.setAnnotationMode('polygon');
        }

        // Extract images from the results folder
        const resultsFolder = loadedZip.folder('results');
        if (resultsFolder) {
          // Get only files directly in the results folder
          for (const [relativePath, file] of Object.entries(resultsFolder.files)) {
            if (!file.dir && 
                relativePath.startsWith('results/') && 
                relativePath.split('/').length === 2 && 
                relativePath.match(/\.(jpg|jpeg|png|gif)$/i)) {
              const blob = await file.async('blob');
              const imageUrl = URL.createObjectURL(blob);
              this.resultImages.push(imageUrl);
            }
          }
        }
        
        this.snackBar.open('Annotations submitted successfully!', 'Close', {
          duration: 3000
        });

        // Wait for the gallery to be rendered
        setTimeout(() => {
          this.scrollToGallery();
        }, 100);
      }
    } catch (error) {
      console.error('Error in submitAnnotations:', error);
      this.snackBar.open('Error submitting annotations', 'Close', {
        duration: 3000
      });
    } finally {
      this.isLoading = false;
    }
  }

  private clearResultImages(): void {
    // Clean up existing object URLs
    this.resultImages.forEach(url => URL.revokeObjectURL(url));
    this.resultImages = [];
  }

  ngOnDestroy(): void {
    // Clean up any remaining object URLs
    this.clearResultImages();
  }

  goBack(): void {
    this.router.navigate(['/']);
  }

  private getImageId(image: File | null): string {
    return image ? `${image.name}-${image.lastModified}` : '';
  }

  validateSplitValue(value: number): void {
    if (value < 0) this.splitValue = 0;
    if (value > 1) this.splitValue = 1;
  }

  closeGallery(): void {
    this.showGallery = false;
    // Clean up image URLs
    this.resultImages.forEach(url => URL.revokeObjectURL(url));
    this.resultImages = [];
  }

  private scrollToGallery(): void {
    if (this.gallerySection && this.resultImages.length > 0) {
      this.gallerySection.nativeElement.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  }

  onClassSelected(category: CocoCategory): void {
    console.log('Class selected:', category);
    this.selectedCategory = category;
    this.updateAnnotationsForCurrentImage();
  }

  private updateAnnotationsForCurrentImage(): void {
    if (!this.selectedCategory || !this.currentCocoImageId) {
      console.log('Cannot update annotations - missing category or image ID:', {
        selectedCategory: this.selectedCategory,
        currentCocoImageId: this.currentCocoImageId
      });
      this.currentAnnotations = [];
      return;
    }

    console.log('Updating annotations with:', {
      selectedCategory: this.selectedCategory,
      currentCocoImageId: this.currentCocoImageId,
      selectedImage: this.selectedImage?.name
    });

    const annotations = this.cocoDatasetService.getAnnotationsForClass(
      this.currentCocoImageId,
      this.selectedCategory.id
    );

    // Update the annotations that will be passed to components
    this.currentAnnotations = annotations;
    console.log('Updated currentAnnotations:', {
      count: this.currentAnnotations.length,
      annotations: this.currentAnnotations.map(ann => ({
        id: ann.id,
        bbox: ann.bbox
      }))
    });
  }

  async onBoundingBoxDrawn(bbox: number[]): Promise<void> {
    if (!this.currentCocoImageId) {
      this.snackBar.open('No image selected', 'Close', { duration: 3000 });
      return;
    }

    // Open class selection dialog
    const dialogRef = this.dialog.open(ClassSelectionDialogComponent, {
      data: { categories: this.categories }
    });

    const result = await dialogRef.afterClosed().toPromise();
    if (!result) return; // Dialog was cancelled

    let categoryId: number;

    if (result.type === 'new') {
      // Create new category
      const newCategory = this.cocoDatasetService.addCategory(
        result.name,
        result.supercategory
      );
      categoryId = newCategory.id;
      this.categories = this.cocoDatasetService.getCategories();
    } else {
      categoryId = result.categoryId;
    }

    // Create a new COCO annotation
    const newAnnotation: CocoAnnotation = {
      id: Date.now(),
      image_id: this.currentCocoImageId,
      category_id: categoryId,
      bbox: bbox,
      iscrowd: 0
    };

    // Add the annotation to the dataset
    this.cocoDatasetService.addAnnotation(newAnnotation);
    
    // Update the current annotations if the selected category matches
    if (this.selectedCategory?.id === categoryId) {
      this.updateAnnotationsForCurrentImage();
    } else {
      // Select the new category
      const category = this.cocoDatasetService.getCategoryById(categoryId);
      if (category) {
        this.selectedCategory = category;
        this.updateAnnotationsForCurrentImage();
      }
    }

    this.snackBar.open('Bounding box added successfully', 'Close', { duration: 3000 });
  }

  async onBoundingBoxRemoved(annotation: CocoAnnotation): Promise<void> {
    console.log('Before removal - currentAnnotations:', this.currentAnnotations.length);
    
    // Remove from current annotations array
    this.currentAnnotations = this.currentAnnotations.filter(ann => {
      const shouldKeep = ann.id !== annotation.id;
      if (!shouldKeep) {
        console.log('Removing annotation:', ann);
      }
      return shouldKeep;
    });

    // Remove from dataset
    const dataset = this.cocoDatasetService.getCurrentDataset();
    if (dataset) {
      dataset.annotations = dataset.annotations.filter(ann => ann.id !== annotation.id);
      this.cocoDatasetService.setDataset(dataset);
    }

    console.log('After removal - currentAnnotations:', this.currentAnnotations.length);

    // Update annotations for current image
    this.updateAnnotationsForCurrentImage();

    this.snackBar.open('Bounding box removed', 'Close', { duration: 3000 });
  }

  get canDownload(): boolean {
    const dataset = this.cocoDatasetService.getCurrentDataset();
    return this.uploadedFiles.length > 0 && dataset !== null;
  }

  async downloadAnnotations(): Promise<void> {
    const dataset = this.cocoDatasetService.getCurrentDataset();
    if (!dataset || this.uploadedFiles.length === 0) {
      this.snackBar.open('No annotations available to download', 'Close', { duration: 3000 });
      return;
    }

    const formData = new FormData();
     
    // Add all original uploaded images
    this.uploadedFiles.forEach(file => {
      formData.append('images', file);
    });
     
    // Add the current COCO dataset with annotations
    formData.append('annotations', JSON.stringify(dataset));

    try {
      this.isLoading = true;
      const response = await this.http.post(`http://${config.IP}:${config.PORT}/update_annotations`, formData, {
        responseType: 'blob'
      }).toPromise();

      if (response) {
        // Create and trigger download
        const blob = new Blob([response], { type: 'application/zip' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'annotations.zip';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);

        this.snackBar.open('Download started successfully', 'Close', { duration: 3000 });
      }
    } catch (error) {
      console.error('Error downloading annotations:', error);
      this.snackBar.open('Error downloading annotations', 'Close', { duration: 3000 });
    } finally {
      this.isLoading = false;
    }
  }
} 
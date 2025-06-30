import { Component, ElementRef, EventEmitter, Input, OnChanges, OnInit, AfterViewInit, OnDestroy, Output, SimpleChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTooltipModule } from '@angular/material/tooltip';
import { AnnotationService, ClickAnnotation, PolygonAnnotation } from '../../services/annotation.service';
import { CocoAnnotation, CocoDatasetService } from '../../services/coco-dataset.service';
import { HttpClient } from '@angular/common/http';
import { MatDialog } from '@angular/material/dialog';
import { ClassSelectionDialogComponent } from '../class-selection-dialog/class-selection-dialog.component';
import { config } from '../../../../src/config';
@Component({
  selector: 'app-image-canvas',
  standalone: true,
  imports: [
    CommonModule,
    MatTooltipModule
  ],
  template: `
    <div class="canvas-container" #container style="display: flex; justify-content: center; align-items: center; height: 600px;">
      <div class="canvas-wrapper" >
        <canvas #canvas></canvas>
        
        <!-- Annotation dots -->
        <div
          *ngFor="let dot of currentAnnotations.clickAnnotations; let i = index"
          class="annotation-dot"
          [style.left.px]="scaleX(dot.x) + offsetX"
          [style.top.px]="scaleY(dot.y) + offsetY"
          [matTooltip]="'Click ' + (i + 1)"
        ></div>
        
        <!-- Polygon points -->
        <div
          *ngFor="let point of currentPolygonPoints; let i = index"
          class="annotation-dot"
          [class.positive]="!isCtrlPressed"
          [class.negative]="isCtrlPressed"
          [style.left.px]="scaleX(point.x) + offsetX"
          [style.top.px]="scaleY(point.y) + offsetY"
          [matTooltip]="'Point ' + (i + 1)"
        ></div>

        <!-- Bounding boxes -->
        <div
          *ngFor="let annotation of boundingBoxes; let i = index"
          class="bounding-box"
          [class.hovered]="hoveredBoxIndex === i"
          [style.left.px]="scaleX(annotation.bbox?.[0] || 0) + offsetX"
          [style.top.px]="scaleY(annotation.bbox?.[1] || 0) + offsetY"
          [style.width.px]="scaleWidth(annotation.bbox?.[2] || 0)"
          [style.height.px]="scaleHeight(annotation.bbox?.[3] || 0)"
          (mouseenter)="onBoxHover(i)"
          (mouseleave)="onBoxLeave()"
          (click)="onBoxClick(i)"
          [matTooltip]="'Click to remove the bounding box'"
        ></div>
        
        <!-- Eraser preview -->
        <div
          *ngIf="annotationMode === 'eraser'"
          class="eraser-preview"
          [style.left.px]="currentX"
          [style.top.px]="currentY"
          [style.width.px]="eraserSize"
          [style.height.px]="eraserSize"
        ></div>
      </div>
    </div>
  `,
  styles: [`
    .canvas-container {
      width: 100%;
      overflow: auto;
      background: #f5f5f5;
      border: 1px solid #ddd;
      border-radius: 4px;
    }

    .canvas-wrapper {
      position: relative;
      margin: 0 auto;
      height: 100%;
    }

    canvas {
      display: block;
      max-height: 100%;
      max-width: 100%;
      object-fit: contain;
    }

    .eraser-preview {
      position: absolute;
      border: 2px solid rgba(255, 0, 0, 0.5);
      background: rgba(255, 0, 0, 0.1);
      border-radius: 50%;
      pointer-events: none;
      transform: translate(-50%, -50%);
    }

    .annotation-dot {
      position: absolute;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
      background-color: #2196F3;
      border: 2px solid #1976D2;
    }

    .annotation-dot.positive {
      background-color: rgba(0, 255, 0, 0.5);
      border: 1px solid rgba(0, 200, 0, 0.8);
    }

    .annotation-dot.negative {
      background-color: rgba(255, 0, 0, 0.5);
      border: 1px solid rgba(200, 0, 0, 0.8);
    }

    .bounding-box {
      position: absolute;
      border: 2px solid #2196F3;
      background-color: rgba(33, 150, 243, 0.1);
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .bounding-box.hovered {
      border: 2px solid #1565C0;
      background-color: rgba(33, 150, 243, 0.3);
      box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.5);
    }
  `]
})
export class ImageCanvasComponent implements OnChanges, OnInit, AfterViewInit, OnDestroy {
  @Input() selectedImage: File | null = null;
  @Input() annotationMode: 'text' | 'click' | 'eraser' | 'polygon' = 'click';
  @Input() eraserSize = 20;
  @Input() annotations: CocoAnnotation[] = [];
  @Input() isLoading = false;
  @Output() annotationAdded = new EventEmitter<any>();
  @Output() annotationRemoved = new EventEmitter<any>();
  @Output() loadingChange = new EventEmitter<boolean>();

  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('container', { static: true }) containerRef!: ElementRef<HTMLDivElement>;

  private ctx!: CanvasRenderingContext2D;
  private image: HTMLImageElement | null = null;
  
  public isDrawing = false;
  public currentX = 0;
  public currentY = 0;
  public canvasWidth = 0;
  public canvasHeight = 0;
  public boundingBoxes: CocoAnnotation[] = [];
  
  public currentAnnotations = {
    clickAnnotations: [] as ClickAnnotation[],
    polygons: [] as PolygonAnnotation[]
  };

  // Polygon-specific properties
  public currentPolygonPoints: { x: number; y: number }[] = [];
  public isCtrlPressed = false;
  private currentPolygonIndex = -1;

  private currentImageId = '';
  private boundStartDrawing: (e: MouseEvent) => void;
  private boundDraw: (e: MouseEvent) => void;
  private boundStopDrawing: () => void;
  private boundHandleClick: (e: MouseEvent) => void;
  private boundHandleKeyDown: (e: KeyboardEvent) => void;
  private boundHandleKeyUp: (e: KeyboardEvent) => void;
  private boundHandleMouseMove: (e: MouseEvent) => void;

  private displayScale = 1;
  public offsetX = 0;
  public offsetY = 0;

  public hoveredBoxIndex = -1;

  constructor(
    private annotationService: AnnotationService,
    private http: HttpClient,
    private dialog: MatDialog,
    private cocoDatasetService: CocoDatasetService
  ) {
    this.boundStartDrawing = this.startDrawing.bind(this);
    this.boundDraw = this.draw.bind(this);
    this.boundStopDrawing = this.stopDrawing.bind(this);
    this.boundHandleClick = this.handleClick.bind(this);
    this.boundHandleKeyDown = this.handleKeyDown.bind(this);
    this.boundHandleKeyUp = this.handleKeyUp.bind(this);
    this.boundHandleMouseMove = this.handleMouseMove.bind(this);
  }

  ngOnInit(): void {
    const context = this.canvasRef.nativeElement.getContext('2d');
    if (!context) {
      throw new Error('Could not get canvas context');
    }
    this.ctx = context;
    this.setupEventListeners();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['selectedImage'] && this.selectedImage) {
      const imageId = this.cocoDatasetService.getImageIdByFilename(this.selectedImage.name);
      if (imageId) {
        this.currentImageId = imageId.toString();
      }
      
      // Load image annotations
      this.currentAnnotations = {
        clickAnnotations: [],
        polygons: []
      };
      
      const savedAnnotations = this.annotationService.getAnnotations(this.currentImageId);
      this.currentAnnotations.clickAnnotations = [...savedAnnotations.clickAnnotations];
      this.currentAnnotations.polygons = [...savedAnnotations.polygons];
      
      // Reset polygon state
      this.currentPolygonPoints = [];
      this.currentPolygonIndex = -1;
      
      // Load the new image
      this.loadImage();
    }

    if (changes['annotations']) {
      console.log('Annotations changed:', this.annotations);
      // Update boundingBoxes when annotations input changes
      this.boundingBoxes = [...this.annotations];
      // Force redraw
      this.redrawAnnotations();
    }

    if (changes['annotationMode']) {
      // Reset polygon state when switching modes
      if (this.annotationMode !== 'polygon' && this.currentPolygonPoints.length > 0) {
        this.currentPolygonPoints = [];
        this.redrawAnnotations();
      }
    }
  }

  ngAfterViewInit(): void {
    if (this.selectedImage) {
      this.loadImage();
    }
  }

  private handleMouseMove(e: MouseEvent): void {
    const { x, y } = this.getCanvasCoordinates(e);
    this.currentX = x;
    this.currentY = y;

    // Redraw if we're in polygon mode and have points
    if (this.annotationMode === 'polygon' && this.currentPolygonPoints.length > 0) {
      this.redrawAnnotations();
    }
  }

  private setupEventListeners(): void {
    const canvas = this.canvasRef.nativeElement;

    canvas.addEventListener('mousedown', this.boundStartDrawing);
    canvas.addEventListener('mousemove', this.boundDraw);
    canvas.addEventListener('mousemove', this.boundHandleMouseMove);
    canvas.addEventListener('mouseup', this.boundStopDrawing);
    canvas.addEventListener('mouseleave', this.boundStopDrawing);
    canvas.addEventListener('click', this.boundHandleClick);
    
    document.addEventListener('keydown', this.boundHandleKeyDown);
    document.addEventListener('keyup', this.boundHandleKeyUp);
  }

  ngOnDestroy(): void {
    const canvas = this.canvasRef.nativeElement;
    canvas.removeEventListener('mousedown', this.boundStartDrawing);
    canvas.removeEventListener('mousemove', this.boundDraw);
    canvas.removeEventListener('mousemove', this.boundHandleMouseMove);
    canvas.removeEventListener('mouseup', this.boundStopDrawing);
    canvas.removeEventListener('mouseleave', this.boundStopDrawing);
    canvas.removeEventListener('click', this.boundHandleClick);
    document.removeEventListener('keydown', this.boundHandleKeyDown);
    document.removeEventListener('keyup', this.boundHandleKeyUp);
  }

  private loadImage(): void {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.image = new Image();
      this.image.onload = () => {
        this.setCanvasSize();
        this.drawImage();
        this.redrawAnnotations();
      };
      this.image.src = e.target?.result as string;
    };
    reader.readAsDataURL(this.selectedImage!);
  }

  private redrawAnnotations(): void {
    console.log('Redrawing annotations with boundingBoxes:', this.boundingBoxes);
    this.drawImage();
    
    // Draw all bounding boxes only in polygon mode
    if (this.ctx && this.boundingBoxes && this.annotationMode === 'polygon') {
      this.boundingBoxes.forEach(annotation => {
        if (annotation.bbox) {
          const [x, y, width, height] = annotation.bbox;
          
          // Draw the bounding box
          this.ctx.strokeStyle = '#2196F3';
          this.ctx.lineWidth = 2;
          this.ctx.fillStyle = 'rgba(33, 150, 243, 0.1)';
          
          // Scale coordinates to canvas size
          const scaledX = this.scaleX(x);
          const scaledY = this.scaleY(y);
          const scaledWidth = this.scaleWidth(width);
          const scaledHeight = this.scaleHeight(height);
          
          this.ctx.beginPath();
          this.ctx.rect(scaledX, scaledY, scaledWidth, scaledHeight);
          this.ctx.stroke();
          this.ctx.fill();
        }
      });
    }
  }

  private setCanvasSize(): void {
    if (!this.image) return;

    const container = this.containerRef.nativeElement;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    // Calculate scale to fit image within container while maintaining aspect ratio
    const scaleX = containerWidth / this.image.width;
    const scaleY = containerHeight / this.image.height;
    this.displayScale = Math.min(scaleX, scaleY);

    // Set canvas dimensions
    this.canvasWidth = this.image.width * this.displayScale;
    this.canvasHeight = this.image.height * this.displayScale;
    
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.canvasWidth;
    canvas.height = this.canvasHeight;

    // Calculate offset to center the image
    this.offsetX = (containerWidth - this.canvasWidth) / 2;
    this.offsetY = (containerHeight - this.canvasHeight) / 2;

    // Update canvas style
    canvas.style.marginLeft = `${this.offsetX}px`;
    canvas.style.marginTop = `${this.offsetY}px`;

    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
  }

  private drawImage(): void {
    if (!this.image) return;
    const canvas = this.canvasRef.nativeElement;
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.ctx.drawImage(this.image, 0, 0, canvas.width, canvas.height);
  }

  private handleKeyDown(e: KeyboardEvent): void {
    if (e.key === 'Control') {
      this.isCtrlPressed = true;
    } else if (e.key === 'Escape') {
      if (this.annotationMode === 'polygon' && this.currentPolygonPoints.length > 0) {
        this.currentPolygonPoints = [];
        this.redrawAnnotations();
      }
    } else if (e.key === 'Enter') {
      if (this.annotationMode === 'polygon' && this.currentPolygonPoints.length >= 3) {
        this.completePolygon();
      }
    }
  }

  private handleKeyUp(e: KeyboardEvent): void {
    if (e.key === 'Control') {
      this.isCtrlPressed = false;
    }
  }

  private completePolygon(): void {
    if (this.currentPolygonPoints.length < 3) return;

    const polygon: PolygonAnnotation = {
      points: [...this.currentPolygonPoints]
    };

    if (this.currentPolygonIndex === -1) {
      this.annotationService.addPolygon(this.currentImageId, polygon);
      this.currentAnnotations.polygons.push(polygon);
      this.annotationAdded.emit({
        type: 'polygon',
        points: polygon.points
      });
    } else {
      this.annotationService.updatePolygon(this.currentImageId, this.currentPolygonIndex, polygon);
      this.currentAnnotations.polygons[this.currentPolygonIndex] = polygon;
    }

    this.currentPolygonPoints = [];
    this.currentPolygonIndex = -1;
    this.redrawAnnotations();
  }

  private getCanvasCoordinates(e: MouseEvent): { x: number; y: number } {
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    
    // Get raw click coordinates relative to canvas
    const rawX = e.clientX - rect.left;
    const rawY = e.clientY - rect.top;

    // Convert to image coordinates
    const x = rawX / this.displayScale;
    const y = rawY / this.displayScale;
    
    return { x, y };
  }

  private async handleClick(e: MouseEvent): Promise<void> {
    const coords = this.getCanvasCoordinates(e);
    const x = coords.x;
    const y = coords.y;

    if (this.annotationMode === 'eraser') {
      this.eraseAt(x, y);
      return;
    }

    if (this.annotationMode === 'polygon') {
      if (this.currentPolygonPoints.length > 2) {
        const firstPoint = this.currentPolygonPoints[0];
        const distance = Math.sqrt(Math.pow(firstPoint.x - x, 2) + Math.pow(firstPoint.y - y, 2));
        
        if (distance <= 10) {
          this.completePolygon();
          return;
        }
      }
      
      this.currentPolygonPoints.push({ x, y });
      this.redrawAnnotations();
      return;
    }

    if (this.annotationMode === 'click' && this.selectedImage) {
      const formData = new FormData();
      formData.append('images', this.selectedImage);
      formData.append('coordinates', JSON.stringify({ x: Math.round(x), y: Math.round(y) }));

      try {
        this.loadingChange.emit(true); // Start loading
        const response: any = await this.http.post(`http://${config.IP}:${config.PORT}/annotate`, formData).toPromise();
        console.log('Annotation API response:', response);

        if (response && 'bbox' in response) {
          // Log the bbox data for debugging
          console.log('API bbox data:', response.bbox);
          
          // Open class selection dialog
          const dialogRef = this.dialog.open(ClassSelectionDialogComponent, {
            data: { categories: this.cocoDatasetService.getCategories() }
          });

          const result = await dialogRef.afterClosed().toPromise();
          if (!result) {
            this.loadingChange.emit(false); // Stop loading if dialog is cancelled
            return;
          }

          let categoryId: number;

          if (result.type === 'new') {
            // Create new category
            const newCategory = this.cocoDatasetService.addCategory(
              result.name,
              result.supercategory
            );
            categoryId = newCategory.id;
          } else {
            categoryId = result.categoryId;
          }

          // Create a new COCO annotation
          // Ensure bbox is in COCO format: [x, y, width, height]
          const apiBbox = response.bbox as number[];
          let bbox: number[];
          
          // Check if API is returning [x1, y1, x2, y2] format
          if (apiBbox.length === 4 && apiBbox[2] > apiBbox[0] && apiBbox[3] > apiBbox[1]) {
            // Convert from [x1, y1, x2, y2] to [x, y, width, height]
            bbox = [
              apiBbox[0], // x
              apiBbox[1], // y
              apiBbox[2] - apiBbox[0], // width
              apiBbox[3] - apiBbox[1]  // height
            ];
          } else {
            // Assume it's already in [x, y, width, height] format
            bbox = apiBbox;
          }

          console.log('Converted bbox:', bbox);

          const newAnnotation: CocoAnnotation = {
            id: Date.now(),
            image_id: parseInt(this.currentImageId),
            category_id: categoryId,
            bbox: bbox,
            area: bbox[2] * bbox[3], // width * height
            iscrowd: 0
          };

          // If score is available in the response, add it to the annotation
          if ('score' in response) {
            (newAnnotation as any).score = response.score || 0.25;
          }

          // Add the annotation to the dataset
          this.cocoDatasetService.addAnnotation(newAnnotation);
          
          // Add to local bounding boxes for display
          this.boundingBoxes = [...this.boundingBoxes, newAnnotation];

          // Emit the annotation added event
          this.annotationAdded.emit({
            type: 'click',
            annotation: newAnnotation
          });
        }
      } catch (error) {
        console.error('Error calling annotation API:', error);
      } finally {
        this.loadingChange.emit(false); // Stop loading
      }
    }
  }

  private drawCurrentPolygon(): void {
    if (this.currentPolygonPoints.length < 2) return;

    this.ctx.beginPath();
    this.ctx.moveTo(this.currentPolygonPoints[0].x, this.currentPolygonPoints[0].y);
    
    for (let i = 1; i < this.currentPolygonPoints.length; i++) {
      this.ctx.lineTo(this.currentPolygonPoints[i].x, this.currentPolygonPoints[i].y);
    }

    if (this.currentX && this.currentY) {
      this.ctx.lineTo(this.currentX, this.currentY);
    }

    // Fill with semi-transparent color
    this.ctx.fillStyle = 'rgba(33, 150, 243, 0.2)';  // Light blue with 0.2 opacity
    this.ctx.fill();
    
    // Draw the border
    this.ctx.strokeStyle = '#2196F3';  // Solid blue for border
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
  }

  private drawPolygon(polygon: PolygonAnnotation): void {
    if (polygon.points.length < 2) return;

    this.ctx.beginPath();
    this.ctx.moveTo(polygon.points[0].x, polygon.points[0].y);
    
    for (let i = 1; i < polygon.points.length; i++) {
      this.ctx.lineTo(polygon.points[i].x, polygon.points[i].y);
    }

    this.ctx.closePath();
    
    // Fill with semi-transparent color
    this.ctx.fillStyle = 'rgba(33, 150, 243, 0.2)';  // Light blue with 0.2 opacity
    this.ctx.fill();
    
    // Draw the border
    this.ctx.strokeStyle = '#2196F3';  // Solid blue for border
    this.ctx.lineWidth = 2;
    this.ctx.stroke();
  }

  // Change from private to public for template access
  public scaleX(x: number): number {
    if (!this.image) return 0;
    return x * this.displayScale;
  }

  public scaleY(y: number): number {
    if (!this.image) return 0;
    return y * this.displayScale;
  }

  public scaleWidth(width: number): number {
    if (!this.image) return 0;
    return width * this.displayScale;
  }

  public scaleHeight(height: number): number {
    if (!this.image) return 0;
    return height * this.displayScale;
  }

  private eraseAt(x: number, y: number): void {
    const radius = this.eraserSize / 2;
    let needsRedraw = false;

    // Erase polygons
    const remainingPolygons = this.currentAnnotations.polygons.filter(polygon => {
      // Check if any point of the polygon is within eraser radius
      for (const point of polygon.points) {
        const distance = Math.sqrt(Math.pow(point.x - x, 2) + Math.pow(point.y - y, 2));
        if (distance <= radius) {
          needsRedraw = true;
          return false; // Remove this polygon
        }
      }
      return true;
    });

    if (this.currentAnnotations.polygons.length !== remainingPolygons.length) {
      this.currentAnnotations.polygons = remainingPolygons;
      const annotations = this.annotationService.getAnnotations(this.currentImageId);
      annotations.polygons = [...remainingPolygons];
      
      this.annotationRemoved.emit({
        type: 'polygon',
      //imageId: this.currentImageId,
        polygons: remainingPolygons
      });
    }

    // Erase click annotations
    const remainingClicks = this.currentAnnotations.clickAnnotations.filter(click => {
      const distance = Math.sqrt(Math.pow(click.x - x, 2) + Math.pow(click.y - y, 2));
      if (distance <= radius) {
        needsRedraw = true;
        return false;
      }
      return true;
    });

    if (this.currentAnnotations.clickAnnotations.length !== remainingClicks.length) {
      this.currentAnnotations.clickAnnotations = remainingClicks;
      const annotations = this.annotationService.getAnnotations(this.currentImageId);
      annotations.clickAnnotations = [...remainingClicks];
      
        this.annotationRemoved.emit({
          type: 'click',
          //imageId: this.currentImageId,
        clicks: remainingClicks
        });
      }

    if (needsRedraw) {
      this.redrawAnnotations();
    }
  }

  private startDrawing(e: MouseEvent): void {
    if (this.annotationMode === 'eraser') {
    this.isDrawing = true;
    const { x, y } = this.getCanvasCoordinates(e);
    this.currentX = x;
    this.currentY = y;
      this.eraseAt(x, y);
    }
  }

  private draw(e: MouseEvent): void {
    if (!this.isDrawing || this.annotationMode !== 'eraser') return;
    
    const { x, y } = this.getCanvasCoordinates(e);
    this.currentX = x;
    this.currentY = y;
      this.eraseAt(x, y);
  }

  private stopDrawing(): void {
    this.isDrawing = false;
  }

  public onBoxHover(index: number): void {
    this.hoveredBoxIndex = index;
  }

  public onBoxLeave(): void {
    this.hoveredBoxIndex = -1;
  }

  public onBoxClick(index: number): void {
    const removedBox = this.boundingBoxes[index];
    if (removedBox) {
      // Remove from local boundingBoxes immediately
      this.boundingBoxes = this.boundingBoxes.filter((_, i) => i !== index);
      
      // Force redraw
      this.redrawAnnotations();

      // Remove from dataset service
      const dataset = this.cocoDatasetService.getCurrentDataset();
      if (dataset) {
        dataset.annotations = dataset.annotations.filter(ann => ann.id !== removedBox.id);
        this.cocoDatasetService.setDataset(dataset);
      }

      // Emit the removed box
      this.annotationRemoved.emit(removedBox);
    }
  }
} 
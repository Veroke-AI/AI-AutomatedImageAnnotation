import { Component, Input, Output, EventEmitter, OnChanges, SimpleChanges, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CocoAnnotation } from '../../services/coco-dataset.service';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatTooltipModule } from '@angular/material/tooltip';

@Component({
  selector: 'app-polygon-drawer',
  template: `
    <div class="polygon-drawer-container" #container style="display: flex; justify-content: center; align-items: center; height: 600px;">
      <div class="canvas-wrapper">
        <canvas #canvas
          [width]="imageWidth"
          [height]="imageHeight"
          [style.margin-left.px]="offsetX"
          [style.margin-top.px]="offsetY"
          (mousedown)="onMouseDown($event)"
          (mousemove)="onMouseMove($event)"
          (mouseup)="onMouseUp($event)">
        </canvas>

        <!-- Bounding boxes -->
        <div
          *ngFor="let annotation of annotations; let i = index"
          class="bounding-box"
          [class.hovered]="hoveredBoxIndex === i"
          [style.left.px]="(annotation.bbox?.[0] || 0) * scale + offsetX"
          [style.top.px]="(annotation.bbox?.[1] || 0) * scale + offsetY"
          [style.width.px]="(annotation.bbox?.[2] || 0) * scale"
          [style.height.px]="(annotation.bbox?.[3] || 0) * scale"
          (mouseenter)="onBoxHover(i)"
          (mouseleave)="onBoxLeave()"
          (click)="onBoxClick(i)"
          [matTooltip]="'Click to remove the bounding box'"
        ></div>
      </div>
    </div>
  `,
  styles: [`
    .polygon-drawer-container {
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
      cursor: crosshair;
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
  `],
  standalone: true,
  imports: [CommonModule, MatButtonModule, MatTooltipModule]
})
export class PolygonDrawerComponent implements OnChanges, AfterViewInit {
  @Input() imageFile: File | null = null;
  @Input() annotations: CocoAnnotation[] = [];
  @Input() selectedCategoryId: number | null = null;
  @Input() annotationMode: 'click' | 'polygon' = 'click';
  @Output() boundingBoxDrawn = new EventEmitter<number[]>();
  @Output() boundingBoxRemoved = new EventEmitter<CocoAnnotation>();

  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('container', { static: true }) containerRef!: ElementRef<HTMLDivElement>;

  private image: HTMLImageElement | null = null;
  imageWidth = 800;
  imageHeight = 600;
  public offsetX = 0;
  public offsetY = 0;
  public scale = 1;

  // Bounding box drawing state
  private isDrawing = false;
  private startX = 0;
  private startY = 0;
  private currentX = 0;
  private currentY = 0;
  private ctx: CanvasRenderingContext2D | null = null;

  public hoveredBoxIndex: number = -1;

  constructor() {
    console.log('PolygonDrawerComponent constructed');
  }

  ngAfterViewInit() {
    this.ctx = this.canvasRef.nativeElement.getContext('2d');
    if (this.imageFile) {
      this.loadImage(this.imageFile);
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['imageFile'] && this.imageFile) {
      this.loadImage(this.imageFile);
    }
    
    if (this.image && (changes['annotations'] || changes['selectedCategoryId'])) {
      this.draw();
    }
  }

  private loadImage(file: File) {
    const objectUrl = URL.createObjectURL(file);
    const newImage = new Image();
    
    newImage.onerror = (error) => {
      console.error('Error loading image:', error);
      URL.revokeObjectURL(objectUrl);
    };
    
    newImage.onload = () => {
      URL.revokeObjectURL(objectUrl);
      this.image = newImage;
      this.setupCanvas();
      requestAnimationFrame(() => {
        this.setupCanvas(); // Call twice to ensure proper sizing
        this.draw();
      });
    };
    
    newImage.src = objectUrl;
  }

  private setupCanvas() {
    if (!this.image || !this.canvasRef || !this.containerRef) return;

    const canvas = this.canvasRef.nativeElement;
    const container = this.containerRef.nativeElement;
    
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    // Calculate scale to fit image within container while maintaining aspect ratio
    const scaleX = containerWidth / this.image.naturalWidth;
    const scaleY = containerHeight / this.image.naturalHeight;
    this.scale = Math.min(scaleX, scaleY);

    // Set canvas dimensions
    this.imageWidth = Math.floor(this.image.naturalWidth * this.scale);
    this.imageHeight = Math.floor(this.image.naturalHeight * this.scale);

    canvas.width = this.imageWidth;
    canvas.height = this.imageHeight;

    // Calculate offset to center the image
    this.offsetX = (containerWidth - this.imageWidth) / 2;
    this.offsetY = (containerHeight - this.imageHeight) / 2;

    // Update canvas style to ensure proper display
    canvas.style.width = `${this.imageWidth}px`;
    canvas.style.height = `${this.imageHeight}px`;
    canvas.style.marginLeft = `${this.offsetX}px`;
    canvas.style.marginTop = `${this.offsetY}px`;

    if (!this.ctx) {
      this.ctx = canvas.getContext('2d');
    }
    
    if (this.ctx) {
      this.ctx.imageSmoothingEnabled = true;
      this.ctx.imageSmoothingQuality = 'high';
    }
  }

  onMouseDown(event: MouseEvent) {
    if (!this.ctx || !this.image) return;
    
    const point = this.getCanvasPoint(event);
    this.isDrawing = true;
    this.startX = point[0];
    this.startY = point[1];
    this.currentX = point[0];
    this.currentY = point[1];
  }

  onMouseMove(event: MouseEvent) {
    if (!this.isDrawing || !this.ctx || !this.image) return;

    const point = this.getCanvasPoint(event);
    this.currentX = point[0];
    this.currentY = point[1];
    
    // Redraw the canvas and draw the current bounding box
    this.draw();
    this.drawCurrentBox();
  }

  onMouseUp(event: MouseEvent) {
    if (!this.isDrawing || !this.ctx || !this.image) return;

    const point = this.getCanvasPoint(event);
    this.currentX = point[0];
    this.currentY = point[1];

    // Calculate the bounding box in original image coordinates
    const x = Math.min(this.startX, this.currentX) / this.scale;
    const y = Math.min(this.startY, this.currentY) / this.scale;
    const width = Math.abs(this.currentX - this.startX) / this.scale;
    const height = Math.abs(this.currentY - this.startY) / this.scale;

    // Only emit if the box has some size
    if (width > 1 && height > 1) {
      // Round the values to avoid floating point issues
      this.boundingBoxDrawn.emit([
        Math.round(x),
        Math.round(y),
        Math.round(width),
        Math.round(height)
      ]);
    }

    this.isDrawing = false;
    this.draw();
  }

  private getCanvasPoint(event: MouseEvent): number[] {
    const canvas = this.canvasRef.nativeElement;
    const rect = canvas.getBoundingClientRect();
    
    // Calculate the scale of the canvas element vs its internal dimensions
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Convert screen coordinates to canvas coordinates
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    return [x, y];
  }

  private drawCurrentBox() {
    if (!this.ctx) return;

    // Draw the current box being created
    const x = Math.min(this.startX, this.currentX);
    const y = Math.min(this.startY, this.currentY);
    const width = Math.abs(this.currentX - this.startX);
    const height = Math.abs(this.currentY - this.startY);

    // Clear the canvas and redraw
    this.draw();

    // Draw the current box
    this.ctx.strokeStyle = '#2196F3';
    this.ctx.lineWidth = 2;
    this.ctx.fillStyle = 'rgba(33, 150, 243, 0.1)';
    
    this.ctx.beginPath();
    this.ctx.rect(x, y, width, height);
    this.ctx.stroke();
    this.ctx.fill();

    // Draw dimensions
    const text = `${Math.round(width / this.scale)}x${Math.round(height / this.scale)}`;
    this.ctx.fillStyle = '#2196F3';
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 3;
    this.ctx.font = 'bold 12px Arial';
    
    // Position text above the box
    const textY = y > 20 ? y - 5 : y + height + 15;
    this.ctx.strokeText(text, x + 4, textY);
    this.ctx.fillText(text, x + 4, textY);
  }

  draw() {
    if (!this.ctx || !this.image) return;

    // Clear canvas
    this.ctx.clearRect(0, 0, this.imageWidth, this.imageHeight);

    // Draw image
    this.ctx.drawImage(this.image, 0, 0, this.imageWidth, this.imageHeight);

    // Draw annotations immediately after image
    if (this.annotations) {
      this.annotations.forEach(annotation => {
        if (annotation.bbox) {
          const [x, y, width, height] = annotation.bbox;
          
          // Scale coordinates
          const scaledX = x * this.scale;
          const scaledY = y * this.scale;
          const scaledWidth = width * this.scale;
          const scaledHeight = height * this.scale;
          
          this.ctx!.strokeStyle = '#2196F3';
          this.ctx!.lineWidth = 2;
          this.ctx!.fillStyle = 'rgba(33, 150, 243, 0.1)';
          
          this.ctx!.beginPath();
          this.ctx!.rect(scaledX, scaledY, scaledWidth, scaledHeight);
          this.ctx!.stroke();
          this.ctx!.fill();
        }
      });
    }
  }

  public onBoxHover(index: number): void {
    this.hoveredBoxIndex = index;
  }

  public onBoxLeave(): void {
    this.hoveredBoxIndex = -1;
  }

  public onBoxClick(index: number): void {
    this.removeBox(index);
  }

  private removeBox(index: number): void {
    const removedBox = this.annotations[index];
    if (removedBox) {
      // Remove from local annotations immediately
      this.annotations = this.annotations.filter((_, i) => i !== index);
      // Emit the removed box
      this.boundingBoxRemoved.emit(removedBox);
    }
  }
} 
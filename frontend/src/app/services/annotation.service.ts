import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

interface AnnotationPayload {
  image: File;
  folder: File[];
  labels: string[];
  annotations: any[];
  preprocessing: string;
}

export interface ClickAnnotation {
  x: number;
  y: number;
  //type: 'positive' | 'negative';
}

export interface BrushStroke {
  path: { x: number; y: number }[];
  size: number;
}

export interface PolygonAnnotation {
  points: { x: number; y: number }[];
}

export interface ImageAnnotations {
  clickAnnotations: ClickAnnotation[];
  brushStrokes: BrushStroke[];
  polygons: PolygonAnnotation[];
}

@Injectable({
  providedIn: 'root'
})
export class AnnotationService {
  private apiUrl = environment.apiUrl;
  private annotations = new Map<string, ImageAnnotations>();
  private classes: string[] = [];
  private annotatedImages: { [key: string]: string } = {};

  constructor(private http: HttpClient) {}

  submitAnnotations(payload: AnnotationPayload): Observable<any> {
    const formData = new FormData();
    formData.append('image', payload.image);
    
    payload.folder.forEach((file, index) => {
      formData.append(`folder[${index}]`, file);
    });

    formData.append('labels', JSON.stringify(payload.labels));
    formData.append('annotations', JSON.stringify(payload.annotations));
    formData.append('preprocessing', payload.preprocessing);

    return this.http.post(`http://192.168.1.209:8000/annotate`, formData);
  }

  getAnnotatedImages(): Observable<any> {
    return this.http.get(`${this.apiUrl}/annotations`);
  }

  getAnnotations(imageId: string): ImageAnnotations {
    if (!this.annotations.has(imageId)) {
      this.annotations.set(imageId, {
        clickAnnotations: [],
        brushStrokes: [],
        polygons: []
      });
    }
    return this.annotations.get(imageId)!;
  }

  addClickAnnotation(imageId: string, annotation: ClickAnnotation): void {
    const annotations = this.getAnnotations(imageId);
    annotations.clickAnnotations.push(annotation);
  }

  addBrushStroke(imageId: string, stroke: BrushStroke): void {
    const annotations = this.getAnnotations(imageId);
    annotations.brushStrokes.push(stroke);
  }

  addPolygon(imageId: string, polygon: PolygonAnnotation): void {
    const annotations = this.getAnnotations(imageId);
    annotations.polygons.push(polygon);
  }

  updatePolygon(imageId: string, index: number, polygon: PolygonAnnotation): void {
    const annotations = this.getAnnotations(imageId);
    if (index >= 0 && index < annotations.polygons.length) {
      annotations.polygons[index] = polygon;
    }
  }

  clearAnnotations(imageId: string): void {
    this.annotations.delete(imageId);
  }

  clearAllAnnotations(): void {
    this.annotations.clear();
  }

  setClasses(classes: string[]): void {
    this.classes = classes;
  }

  getClasses(): string[] {
    return this.classes;
  }

  updateAnnotatedImage(imageId: string, annotatedImageData: string): void {
    this.annotatedImages[imageId] = annotatedImageData;
  }

  getAnnotatedImage(imageId: string): string | undefined {
    return this.annotatedImages[imageId];
  }
} 
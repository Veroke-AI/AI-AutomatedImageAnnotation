import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export interface CocoCategory {
  id: number;
  name: string;
  supercategory?: string;
}

export interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  segmentation?: number[][];
  bbox?: number[];
  area?: number;
  iscrowd?: number;
}

export interface CocoImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
}

export interface CocoDataset {
  images: CocoImage[];
  annotations: CocoAnnotation[];
  categories: CocoCategory[];
}

@Injectable({
  providedIn: 'root'
})
export class CocoDatasetService {
  private dataset = new BehaviorSubject<CocoDataset | null>(null);
  private currentImageId = new BehaviorSubject<number | null>(null);

  dataset$ = this.dataset.asObservable();
  currentImageId$ = this.currentImageId.asObservable();

  setDataset(dataset: CocoDataset) {
    this.dataset.next(dataset);
  }

  setCurrentImageId(imageId: number) {
    this.currentImageId.next(imageId);
  }

  getCategories(): CocoCategory[] {
    return this.dataset.value?.categories || [];
  }

  getAnnotationsForClass(imageId: number, categoryId: number): CocoAnnotation[] {
    console.log('Getting annotations for:', {
      imageId,
      categoryId,
      totalAnnotations: this.dataset.value?.annotations.length || 0
    });

    const annotations = (this.dataset.value?.annotations || []).filter(
      ann => ann.image_id === imageId && ann.category_id === categoryId && ann.bbox
    );

    console.log('Found annotations:', {
      count: annotations.length,
      annotations: annotations.map(ann => ({
        id: ann.id,
        image_id: ann.image_id,
        category_id: ann.category_id,
        bbox: ann.bbox
      }))
    });

    return annotations;
  }

  addPolygonAnnotation(annotation: CocoAnnotation) {
    const currentDataset = this.dataset.value;
    if (currentDataset) {
      // Generate a unique ID for the new annotation
      const maxId = Math.max(...currentDataset.annotations.map(a => a.id), 0);
      annotation.id = maxId + 1;
      
      // Calculate area if not provided
      if (!annotation.area && annotation.segmentation) {
        annotation.area = this.calculatePolygonArea(annotation.segmentation[0]);
      }
      
      currentDataset.annotations.push(annotation);
      this.dataset.next(currentDataset);
    }
  }

  private calculatePolygonArea(points: number[]): number {
    let area = 0;
    const numPoints = points.length / 2;
    for (let i = 0; i < numPoints; i++) {
      const j = (i + 1) % numPoints;
      area += points[i * 2] * points[j * 2 + 1];
      area -= points[j * 2] * points[i * 2 + 1];
    }
    return Math.abs(area) / 2;
  }

  getCurrentDataset(): CocoDataset | null {
    return this.dataset.value;
  }

  getImageIdByFilename(filename: string): number | null {
    const dataset = this.dataset.value;
    if (!dataset) {
      console.log('No dataset available when looking up image ID');
      return null;
    }

    console.log('Looking up image ID for:', {
      filename,
      availableImages: dataset.images.map(img => ({
        id: img.id,
        file_name: img.file_name
      }))
    });

    const image = dataset.images.find(img => img.file_name === filename);
    console.log('Found image:', image);
    return image ? image.id : null;
  }

  addAnnotation(annotation: CocoAnnotation) {
    const currentDataset = this.dataset.value;
    if (currentDataset) {
      // Generate a unique ID for the new annotation
      const maxId = Math.max(...currentDataset.annotations.map(a => a.id), 0);
      annotation.id = maxId + 1;
      
      // Calculate area if not provided and bbox exists
      if (!annotation.area && annotation.bbox) {
        annotation.area = annotation.bbox[2] * annotation.bbox[3]; // width * height
      }
      
      currentDataset.annotations.push(annotation);
      this.dataset.next(currentDataset);

      console.log('Added new annotation:', {
        id: annotation.id,
        image_id: annotation.image_id,
        category_id: annotation.category_id,
        bbox: annotation.bbox
      });
    }
  }

  addCategory(name: string, supercategory?: string): CocoCategory {
    const currentDataset = this.dataset.value;
    if (!currentDataset) {
      throw new Error('No dataset available');
    }

    // Generate a unique ID for the new category
    const maxId = Math.max(...currentDataset.categories.map(c => c.id), 0);
    const newCategory: CocoCategory = {
      id: maxId + 1,
      name,
      supercategory
    };

    currentDataset.categories.push(newCategory);
    this.dataset.next(currentDataset);

    console.log('Added new category:', newCategory);
    return newCategory;
  }

  getCategoryById(id: number): CocoCategory | undefined {
    return this.dataset.value?.categories.find(c => c.id === id);
  }
} 
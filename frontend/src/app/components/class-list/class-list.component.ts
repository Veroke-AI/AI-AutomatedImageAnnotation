import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CocoCategory } from '../../services/coco-dataset.service';
import { MatSelectionListChange } from '@angular/material/list';
import { MatListModule } from '@angular/material/list';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-class-list',
  template: `
    <div class="class-list-container">
      <h3>Classes</h3>
      <mat-selection-list [multiple]="false" (selectionChange)="onSelectionChange($event)">
        <mat-list-option *ngFor="let category of categories" [value]="category">
          {{ category.name }}
        </mat-list-option>
      </mat-selection-list>
    </div>
  `,
  styles: [`
    .class-list-container {
      padding: 16px;
      background: white;
      border-radius: 4px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h3 {
      margin: 0 0 16px 0;
      color: #333;
    }
    mat-selection-list {
      max-height: 300px;
      overflow-y: auto;
    }
  `],
  standalone: true,
  imports: [
    MatListModule,
    CommonModule
  ]
})
export class ClassListComponent {
  @Input() categories: CocoCategory[] = [];
  @Output() classSelected = new EventEmitter<CocoCategory>();

  onSelectionChange(event: MatSelectionListChange) {
    const selectedCategory = event.options[0].value as CocoCategory;
    this.classSelected.emit(selectedCategory);
  }
} 
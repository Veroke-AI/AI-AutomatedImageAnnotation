import { Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { VisualIntelligenceComponent } from './components/visual-intelligence/visual-intelligence.component';
import { AnnotationWorkspaceComponent } from './components/annotation-workspace/annotation-workspace.component';

export const routes: Routes = [
  {
    path: '',
    component: HomeComponent
  },
  {
    path: 'visual-intelligence',
    component: VisualIntelligenceComponent
  },
  {
    path: 'annotate',
    component: AnnotationWorkspaceComponent
  },
  {
    path: '**',
    redirectTo: ''
  }
]; 
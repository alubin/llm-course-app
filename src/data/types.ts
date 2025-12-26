// Common interfaces used across all day content files

export interface Prerequisite {
  name: string;
  details: string;
}

export interface Technology {
  name: string;
  purpose: string;
}

export interface Module {
  id: string;
  title: string;
  content: string;
}

export interface Task {
  id: string;
  title: string;
  description: string;
  content: string;
}

export interface Section {
  id: string;
  title: string;
  estimatedTime: string;
  modules?: Module[];
  tasks?: Task[];
}

export interface DayContent {
  id: number;
  title: string;
  subtitle: string;
  duration: string;
  difficulty: string;
  objectives: string[];
  prerequisites: Prerequisite[];
  technologies: Technology[];
  sections: Section[];
}

export interface PythonSection {
  id: string;
  title: string;
  content: string;
}

export interface PythonPrimerContent {
  title: string;
  subtitle: string;
  duration: string;
  sections: PythonSection[];
}

export interface CourseDay {
  id: number;
  title: string;
  description: string;
  icon: string;
  language: string;
  duration: string;
  project: string;
  topics: string[];
  status: 'available' | 'coming-soon';
}

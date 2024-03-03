import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import {getStorage} from 'firebase/storage'
import { getDatabase } from "firebase/database";

export const firebaseConfig = {
    apiKey: "AIzaSyBNMei7IuS4nFKFuSJE27qeIl6yehUCZIE",
    authDomain: "langdesign.firebaseapp.com",
    databaseURL: "https://langdesign-default-rtdb.europe-west1.firebasedatabase.app",
    projectId: "langdesign",
    storageBucket: "langdesign.appspot.com",
    messagingSenderId: "1023431971683",
    appId: "1:1023431971683:web:c5f7805c4c8ef61f707956",
    measurementId: "G-Y5M7JKZ742",
  };

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const storage = getStorage(app);
export const database = getDatabase(app);

import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyDY2sABGhgLiNMUW5_bYw968rH6WuGi2Ag",
  authDomain: "pillmotion-971b6.firebaseapp.com",
  projectId: "pillmotion-971b6",
  storageBucket: "pillmotion-971b6.appspot.com",
  messagingSenderId: "426710909862",
  appId: "1:426710909862:web:85334453c4b3092c224567",
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
